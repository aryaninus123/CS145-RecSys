import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pyspark.sql import functions as sf

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import train_test_split_edges
from sklearn.metrics import roc_auc_score, accuracy_score
from sim4rec.utils import pandas_to_spark

import numpy as np
import pandas as pd
import random

def build_bipartite_graph(users_df, items_df, interactions_df, n_user_features, n_item_features,
                          total_num_users=None, total_num_items=None):
    """
    Builds a PyTorch Geometric Data object for a bipartite user-item graph.
    """
    # Convert Spark DataFrames to Pandas DataFrames if they are not already
    if not isinstance(users_df, pd.DataFrame):
        print("Converting users_df from Spark DataFrame to Pandas DataFrame...")
        users_df_pd = users_df.toPandas()
    else:
        users_df_pd = users_df

    if not isinstance(items_df, pd.DataFrame):
        print("Converting items_df from Spark DataFrame to Pandas DataFrame...")
        items_df_pd = items_df.toPandas()
    else:
        items_df_pd = items_df

    if not isinstance(interactions_df, pd.DataFrame):
        print("Converting interactions_df from Spark DataFrame to Pandas DataFrame...")
        interactions_df_pd = interactions_df.toPandas()
    else:
        interactions_df_pd = interactions_df

    # Determine num_users and num_items for graph indexing
    num_users = total_num_users if total_num_users is not None else users_df_pd["user_idx"].max() + 1
    num_items = total_num_items if total_num_items is not None else items_df_pd["item_idx"].max() + 1

    # Combine user and item features
    max_features_dim = max(n_user_features, n_item_features)
    
    # Create empty feature 
    all_user_features = torch.zeros(num_users, max_features_dim, dtype=torch.float)
    all_item_features = torch.zeros(num_items, max_features_dim, dtype=torch.float)

    # Fill in features for users/items present in the provided dataframes
    user_indices_with_features = users_df_pd["user_idx"].values
    user_attrs = torch.tensor(users_df_pd[[f"user_attr_{i}" for i in range(n_user_features)]].values, dtype=torch.float)
    if user_attrs.shape[1] < max_features_dim: # Pad if needed before assigning
        user_attrs = F.pad(user_attrs, (0, max_features_dim - user_attrs.shape[1]), 'constant', 0)
    all_user_features[user_indices_with_features] = user_attrs

    item_indices_with_features = items_df_pd["item_idx"].values
    item_attrs = torch.tensor(items_df_pd[[f"item_attr_{i}" for i in range(n_item_features)]].values, dtype=torch.float)
    if item_attrs.shape[1] < max_features_dim: # Pad if needed before assigning
        item_attrs = F.pad(item_attrs, (0, max_features_dim - item_attrs.shape[1]), 'constant', 0)
    all_item_features[item_indices_with_features] = item_attrs

    x = torch.cat([all_user_features, all_item_features], dim=0)

    # User-item interactions
    src_nodes = interactions_df_pd["user_idx"].values
    dst_nodes = interactions_df_pd["item_idx"].values + num_users # Offset item indices

    # Create edges for user -> item
    edge_index_u_i = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)

    # Create edges for item -> user
    edge_index_i_u = torch.tensor([dst_nodes, src_nodes], dtype=torch.long)

    # Combine both directions
    edge_index = torch.cat([edge_index_u_i, edge_index_i_u], dim=1)

    data = Data(x=x, edge_index=edge_index)
    data.num_users = num_users
    data.num_items = num_items
    data.num_nodes = num_users + num_items # Total number of nodes
    data.original_edge_index = torch.cat([edge_index_u_i, edge_index_i_u], dim=1)

    return data

class GCNLinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, decoder_type='dot_product'):
        """
        Initializes a GCN-based link prediction model.
        """
        super(GCNLinkPredictor, self).__init__()
        self.conv_layers = torch.nn.ModuleList()
        self.decoder_type = decoder_type

        # Input layer
        self.conv_layers.append(GCNConv(in_channels, hidden_channels))
        # Hidden layers
        for _ in range(num_layers - 1):
            self.conv_layers.append(GCNConv(hidden_channels, hidden_channels))
        # Output layer (maps to out_channels for embeddings)
        self.final_conv = GCNConv(hidden_channels, out_channels)

        if decoder_type == 'mlp':
            self.mlp_decoder = torch.nn.Sequential(
                torch.nn.Linear(2 * out_channels, hidden_channels), # Concatenated user and item embeddings
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(hidden_channels, 1) # Output a single logit for binary classification
            )

    def forward(self, x, edge_index):
        """
        Forward pass through the GCN layers.
        """
        for conv in self.conv_layers:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
        
        # Final GCN layer outputs the node embeddings
        x = self.final_conv(x, edge_index)
        return x

    def decode(self, z, edge_label_index):
        """
        Decodes the likelihood of links based on node embeddings and edge labels.
        """
        # Extract embeddings for the source and destination nodes of the edges
        source_embeddings = z[edge_label_index[0]]
        target_embeddings = z[edge_label_index[1]]

        if self.decoder_type == 'dot_product':
            # Dot product between embeddings
            return (source_embeddings * target_embeddings).sum(dim=1)
        elif self.decoder_type == 'mlp':
            # Concatenate and pass through MLP
            combined_embeddings = torch.cat([source_embeddings, target_embeddings], dim=1)
            return self.mlp_decoder(combined_embeddings).squeeze(1)
        else:
            raise ValueError(f"Unknown decoder type: {self.decoder_type}")

def train_link_predictor(model, optimizer, data):
    """
    Trains the link prediction model for one epoch.
    """
    model.train()
    optimizer.zero_grad()

    train_pos_edge_index = data.edge_index
    num_users = data.num_users
    num_items = data.num_items
    
    # Store as canonical (user_idx, item_idx) pairs
    existing_edges = set()
    for i in range(train_pos_edge_index.shape[1]):
        u_node = train_pos_edge_index[0, i].item()
        v_node = train_pos_edge_index[1, i].item()
        
        # Ensure we always get user_idx and item_idx
        if u_node < num_users:
            existing_edges.add((u_node, v_node - num_users))
        else:
            existing_edges.add((v_node, u_node - num_users))
            
    # Count of unique user-item interactions
    num_positive_interactions = train_pos_edge_index.shape[1] // 2 
    num_negative_samples = num_positive_interactions

    train_neg_edge_index_list = []
    sampled_count = 0
    max_attempts = num_negative_samples * 10 # Prevent infinite loop for very dense graphs

    # Sample negative edges for the training set
    while sampled_count < num_negative_samples and max_attempts > 0:
        # Sample a random user and a random item
        u_idx = random.randint(0, num_users - 1)
        i_idx = random.randint(0, num_items - 1)
        
        # Check if this (user, item) pair already exists as a positive interaction
        if (u_idx, i_idx) not in existing_edges:
            train_neg_edge_index_list.append([u_idx, i_idx + num_users]) # user -> item
            train_neg_edge_index_list.append([i_idx + num_users, u_idx]) # item -> user
            sampled_count += 1
        max_attempts -= 1

    if not train_neg_edge_index_list and num_negative_samples > 0:
        train_neg_edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        train_neg_edge_index = torch.tensor(train_neg_edge_index_list, dtype=torch.long).T

    z = model(data.x, data.edge_index)

    # Prepare labels for positive and negative edges
    pos_score = model.decode(z, train_pos_edge_index)
    neg_score = model.decode(z, train_neg_edge_index)

    # Combine positive and negative scores and labels
    scores = torch.cat([pos_score, neg_score], dim=0)
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])], dim=0)

    loss = F.binary_cross_entropy_with_logits(scores, labels.to(scores.device))
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def evaluate_link_predictor(model, data, edge_label_index, edge_label, num_users):
    """
    Evaluates the link prediction model.
    """
    model.eval()
    z = model(data.x, data.edge_index)
    out = model.decode(z, edge_label_index).view(-1).sigmoid() # Apply sigmoid to get probabilities

    # Ensure all predicted values are within [0, 1] for AUC-ROC
    out = torch.clamp(out, min=1e-12, max=1-1e-12)

    # Filter out self-loops or invalid user-item links
    valid_mask = (edge_label_index[0] < num_users) & (edge_label_index[1] >= num_users)

    if valid_mask.sum() == 0:
        print("Warning: No valid user-item links found for evaluation. Skipping metrics.")
        return 0.5, 0.5
    
    # Apply mask
    out_valid = out[valid_mask]
    edge_label_valid = edge_label[valid_mask]

    auc_score = roc_auc_score(edge_label_valid.cpu().numpy(), out_valid.cpu().numpy())
    predicted_labels = (out_valid > 0.5).cpu().numpy()
    true_labels = edge_label_valid.cpu().numpy()
    accuracy = accuracy_score(true_labels, predicted_labels)
    return auc_score, accuracy

class GCNRecommender:
    def __init__(self, hidden_channels=64, out_channels=32, num_epochs=100, learning_rate=0.01, seed=None, spark_session=None):
        """
        Initializes the RecommenderSystem with GCN and training parameters.
        """
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.seed = seed
        self.models = {} 
        self.fitted_data_info = {}

        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
            random.seed(self.seed)

    def fit(self, log, user_features=None, item_features=None):
        """
        Builds the graph, trains and evaluates the GCN-based recommender models.
        """
        # Convert to Pandas for initial processing to determine max indices
        if not isinstance(user_features, pd.DataFrame):
            user_features_pd = user_features.toPandas()
        else:
            user_features_pd = user_features
        if not isinstance(item_features, pd.DataFrame):
            item_features_pd = item_features.toPandas()
        else:
            item_features_pd = item_features
        if not isinstance(log, pd.DataFrame):
            log_pd = log.toPandas()
        else:
            log_pd = log

        # Infer feature dimensions from the provided dataframes
        n_user_features = sum(1 for col in user_features_pd.columns if col.startswith("user_attr_"))
        n_item_features = sum(1 for col in item_features_pd.columns if col.startswith("item_attr_"))

        if n_user_features == 0 or n_item_features == 0:
            raise ValueError("User or item features not found in the provided DataFrames.")

        # Determine the total number of users and items
        total_num_users = max(user_features_pd["user_idx"].max(), log_pd["user_idx"].max()) + 1
        total_num_items = max(item_features_pd["item_idx"].max(), log_pd["item_idx"].max()) + 1


        # Pass total_num_users and total_num_items to build_bipartite_graph
        data = build_bipartite_graph(
            user_features, item_features, log,
            n_user_features, n_item_features,
            total_num_users=total_num_users, total_num_items=total_num_items
        )

        if data.edge_index.shape[1] < 10:
            print("Not enough edges for train/test split. Please ensure sufficient positive interactions.")
            return

        temp_data_for_split = Data(x=data.x, edge_index=data.original_edge_index, num_nodes=data.num_nodes)
        data_split = train_test_split_edges(temp_data_for_split, test_ratio=0.1, val_ratio=0.05)
        
        data.edge_index = data_split.train_pos_edge_index 
        
        # P
        # repare edge_label_index and edge_label for evaluation sets
        val_edge_label_index = torch.cat([data_split.val_pos_edge_index, data_split.val_neg_edge_index], dim=1)
        val_edge_label = torch.cat([torch.ones(data_split.val_pos_edge_index.shape[1]), 
                                     torch.zeros(data_split.val_neg_edge_index.shape[1])], dim=0)

        test_edge_label_index = torch.cat([data_split.test_pos_edge_index, data_split.test_neg_edge_index], dim=1)
        test_edge_label = torch.cat([torch.ones(data_split.test_pos_edge_index.shape[1]), 
                                      torch.zeros(data_split.test_neg_edge_index.shape[1])], dim=0)


        # Store critical info about the fitted graph for later use in predict
        self.fitted_data_info = {
            'n_user_features': n_user_features,
            'n_item_features': n_item_features,
            'num_users': total_num_users,
            'num_items': total_num_items,
        }

        input_feature_dim = data.x.shape[1]
        
        # GCN with a Small MLP Decoder
        model = GCNLinkPredictor(input_feature_dim, self.hidden_channels, self.out_channels, num_layers=2, decoder_type='mlp')
        optimizer3 = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        self.models['gcn_mlp_decoder'] = model

        for epoch in range(1, self.num_epochs + 1):
            loss = train_link_predictor(model, optimizer3, data)
            if epoch % 10 == 0:
                val_auc, val_acc = evaluate_link_predictor(model, data, val_edge_label_index, val_edge_label, data.num_users)
                print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Val AUC: {val_auc:.4f}, Val Acc: {val_acc:.4f}")
        
        evaluate_link_predictor(model, data, test_edge_label_index, test_edge_label, data.num_users)


    def predict(self, log, k, users, items, user_features, item_features, filter_seen_items=True, model_name='gcn_mlp_decoder'):
        """
        Generates top-K recommendations for specified users, maximizing expected revenue.
        """
        model_name = 'gcn_mlp_decoder' 
        
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not trained. Call fit() first or ensure the model is correctly trained.")
        if not self.fitted_data_info:
             raise RuntimeError("RecommenderSystem has not been fitted yet. Call fit() before predict().")

        model = self.models[model_name]
        model.eval()

        if not isinstance(item_features, pd.DataFrame):
            items_features_pd = item_features.toPandas()
        else:
            items_features_pd = item_features
            
        # Create a mapping from item_idx to price
        item_price_map = items_features_pd.set_index('item_idx')['price'].to_dict()

        # Infer feature dimensions
        if not isinstance(user_features, pd.DataFrame):
            temp_user_features_pd = user_features.limit(1).toPandas()
        else:
            temp_user_features_pd = user_features
        n_user_features = sum(1 for col in temp_user_features_pd.columns if col.startswith("user_attr_"))
        n_item_features = sum(1 for col in items_features_pd.columns if col.startswith("item_attr_"))

        if n_user_features == 0 or n_item_features == 0:
            raise ValueError("User or item features not found in the provided DataFrames for prediction. ")

        # Build the graph for prediction using the total number of users and items from fitting,
        current_data = build_bipartite_graph(
            user_features, item_features, log,
            n_user_features, n_item_features,
            total_num_users=self.fitted_data_info['num_users'],
            total_num_items=self.fitted_data_info['num_items']
        )
        
        with torch.no_grad():
            # Generate embeddings based on the current graph structure for prediction
            z = model(current_data.x, current_data.edge_index) 

        recommendations = []

        # Prepare seen items for filtering
        if not isinstance(log, pd.DataFrame):
            log_pd = log.toPandas()
        else:
            log_pd = log

        seen_items_map = {}
        if filter_seen_items:
            for _, row in log_pd.iterrows():
                user_id = row['user_idx']
                item_id = row['item_idx']
                if user_id not in seen_items_map:
                    seen_items_map[user_id] = set()
                seen_items_map[user_id].add(item_id)

        # Determine the user IDs to predict for
        if not isinstance(users, pd.DataFrame):
            users_pd = users.toPandas()
        else:
            users_pd = users
        user_ids_to_predict = users_pd['user_idx'].unique()

        # Determine the item IDs to consider as candidates
        if not isinstance(items, pd.DataFrame):
            items_pd = items.toPandas()
        else:
            items_pd = items
        item_ids_to_consider = items_pd['item_idx'].unique()


        for user_id in user_ids_to_predict:
            candidate_items = list(item_ids_to_consider) # Initially, all items

            if user_id >= current_data.num_users:
                print(f"Warning: User ID {user_id} is out of bounds for the current prediction graph. Skipping recommendations for this user.")
                continue

            if filter_seen_items and user_id in seen_items_map:
                # Filter out items already seen by this user
                candidate_items = [item_id for item_id in candidate_items if item_id not in seen_items_map[user_id]]
            
            if not candidate_items:
                continue

            # Create prediction edges
            user_nodes_for_pred = torch.tensor([user_id] * len(candidate_items), dtype=torch.long)
            item_nodes_in_graph_for_pred = torch.tensor([item_id + current_data.num_users for item_id in candidate_items], dtype=torch.long)
            
            # Filter out candidate items that are out of bounds for the current graph's item nodes
            valid_candidate_mask = (item_nodes_in_graph_for_pred < current_data.num_nodes)
            if not valid_candidate_mask.all():
                user_nodes_for_pred = user_nodes_for_pred[valid_candidate_mask]
                item_nodes_in_graph_for_pred = item_nodes_in_graph_for_pred[valid_candidate_mask]
                candidate_items = [candidate_items[i] for i, valid in enumerate(valid_candidate_mask) if valid]
            
            if not candidate_items:
                continue

            prediction_edges = torch.stack([user_nodes_for_pred, item_nodes_in_graph_for_pred], dim=0)

            # Get raw scores from the model's decoder and apply sigmoid for probabilities
            link_probabilities = model.decode(z, prediction_edges).sigmoid().cpu().detach().numpy()

            # Get prices for candidate items
            candidate_prices = np.array([item_price_map.get(item_id, 0.0) for item_id in candidate_items]) # Default to 0.0 if price is missing

            # Calculate expected revenue
            expected_revenue = candidate_prices * link_probabilities

            # Create a temporary DataFrame to sort and select top-k items
            temp_df = pd.DataFrame({
                'item_idx': candidate_items,
                'link_probability': link_probabilities,
                'expected_revenue': expected_revenue
            })
            # Sort by expected_revenue to maximize revenue
            temp_df = temp_df.sort_values(by='expected_revenue', ascending=False).head(k)

            for _, row in temp_df.iterrows():
                recommendations.append({
                    'user_idx': user_id,
                    'item_idx': int(row['item_idx']),
                    'relevance': row['expected_revenue']
                })

        final_recs_pd = pd.DataFrame(recommendations)
        return pandas_to_spark(final_recs_pd)