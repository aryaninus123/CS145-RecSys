import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from sim4rec.utils import pandas_to_spark


# Cell: Import libraries and set up environment
"""
# Recommender Systems Analysis and Visualization
This notebook performs an exploratory analysis of recommender systems using the Sim4Rec library.
We'll generate synthetic data, compare multiple baseline recommenders, and visualize their performance.
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as sf
from pyspark.sql import DataFrame, Window
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder, StandardScaler
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.types import DoubleType, ArrayType, StructType, StructField, LongType, IntegerType, StringType

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import math
import random
import tqdm

# Set up plotting
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)

# Initialize Spark session
spark = SparkSession.builder \
    .appName("RecSysVisualization") \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "8") \
    .getOrCreate()

# Set log level to warnings only
spark.sparkContext.setLogLevel("WARN")

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Computes scaled dot-product attention.
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf')) # Apply mask
    p_attn = F.softmax(scores, dim=-1)
    return torch.matmul(p_attn, V)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 8):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Learned projections
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)

        # Final output projection
        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        """Applies multi-head attention.
        """
        B, N, _ = Q.size() 

        # Project Q, K, V and split into heads (B, num_heads, N, d_k)
        Q_proj = self.W_Q(Q).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        K_proj = self.W_K(K).view(B, K.size(1), self.num_heads, self.d_k).transpose(1, 2)
        V_proj = self.W_V(V).view(B, V.size(1), self.num_heads, self.d_k).transpose(1, 2)

        if mask is not None:
            # Ensure mask is suitable for broadcasting across heads (B, 1, N, M) -> (B, num_heads, N, M)
            if mask.dim() == 3: # If mask is (B, N, M)
                mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            elif mask.dim() == 2: # If mask is (N, M) for a single sequence
                mask = mask.unsqueeze(0).unsqueeze(1).repeat(B, self.num_heads, 1, 1)
            
        attn_out = scaled_dot_product_attention(Q_proj, K_proj, V_proj, mask)

        # Concatenate heads and apply final linear projection
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, N, self.d_model)
        output = self.W_O(attn_out)

        return output

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention part
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output)) # Add & Norm

        # Feed-forward part
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output)) # Add & Norm
        return x

class SASRecModel(nn.Module):
    """
    Simplified SASRec-style Transformer model for sequential recommendation.
    Predicts the next item in a sequence.
    """
    def __init__(self, num_items: int, max_seq_length: int, embedding_dim: int,
                 num_heads: int, hidden_size: int, feed_forward_size: int,
                 dropout_rate: float, num_blocks: int, padding_idx: int):
        super().__init__()
        self.num_items = num_items
        self.max_seq_length = max_seq_length
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

        # Item embedding layer
        self.item_embedding = nn.Embedding(num_items, embedding_dim, padding_idx=self.padding_idx)

        # Positional embedding layer
        self.position_embedding = nn.Embedding(max_seq_length, embedding_dim)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, feed_forward_size, dropout_rate)
            for _ in range(num_blocks)
        ])

        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(hidden_size)

        # Prediction head
        self.prediction_head = nn.Linear(hidden_size, num_items)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.padding_idx)


    def forward(self, item_seq: torch.Tensor, mask: torch.Tensor):
        """
        item_seq: Tensor of shape (B, max_seq_length), containing item indices.
        mask: Tensor of shape (B, max_seq_length), 1 for real, 0 for padding.
        """
        # Create positional indices (0 to max_seq_length-1)
        positions = torch.arange(self.max_seq_length, dtype=torch.long, device=item_seq.device)
        positions = positions.unsqueeze(0).expand_as(item_seq)

        # Get item and positional embeddings
        item_embs = self.item_embedding(item_seq)
        pos_embs = self.position_embedding(positions)

        # Combine item and positional embeddings
        x = item_embs + pos_embs
        x = self.dropout(x)

        subsequent_mask = torch.triu(torch.ones((self.max_seq_length, self.max_seq_length),
                                                      device=item_seq.device), diagonal=1).bool()
        padding_mask = mask.unsqueeze(1).unsqueeze(2)
        causal_mask = subsequent_mask.unsqueeze(0).unsqueeze(1) 
        combined_mask = (padding_mask | causal_mask)

        for block in self.transformer_blocks:
            x = block(x, combined_mask)

        x = self.norm(x)
        logits = self.prediction_head(x)
        return logits

class TransformerRecommender:
    def __init__(self, seed=42, max_sequence_length=100,
                 item_embedding_size=128,
                 num_heads=8, hidden_size=128, feed_forward_size=256,
                 dropout_rate=0.1, num_blocks=8,
                 learning_rate=0.00005, epochs=30, batch_size=64,
                 user_id_col: str = "user_idx",
                 item_id_col: str = "item_idx",
                 ):
        self.max_sequence_length = max_sequence_length
        self.user_interaction_sequences = defaultdict(list)
        self.item_prices = {}
        self.user_id_col = user_id_col
        self.item_id_col = item_id_col

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.item_embedding_size = item_embedding_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.feed_forward_size = feed_forward_size
        self.dropout_rate = dropout_rate
        self.num_blocks = num_blocks
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self._current_round = 0 
        self._model_initialized = False 

        self.item_feature_vectors_pd = None

    def fit(self, 
        log: DataFrame = None, 
        user_features: DataFrame = None, 
        item_features: DataFrame = None 
       ):
        """
        Train the recommender model based on interaction history.
        """
        round_to_process = self._current_round 
        self._current_round += 1

        print(f"\n--- Training Round {round_to_process} ---")

        # Initialize model and optimizer, criterion
        if not self._model_initialized:
            max_actual_item_id = item_features.select(sf.max(self.item_id_col)).collect()[0][0]
            num_real_items = max_actual_item_id + 1
            self.padding_item_idx = num_real_items
            self.total_num_items = num_real_items + 1

            self.model = SASRecModel(
                num_items=self.total_num_items,
                max_seq_length=self.max_sequence_length,
                embedding_dim=self.item_embedding_size,
                num_heads=self.num_heads,
                hidden_size=self.hidden_size,
                feed_forward_size=self.feed_forward_size,
                dropout_rate=self.dropout_rate,
                num_blocks=self.num_blocks,
                padding_idx=self.padding_item_idx
            ).to(self.device)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
            self.criterion = nn.CrossEntropyLoss(ignore_index=self.padding_item_idx) # Use padding_item_idx to ignore
            self._model_initialized = True

        self.model.train()

        # Update item prices
        self._update_item_prices(item_features)

        # Curate user interaction sequences
        current_interactions_pd = log.toPandas()
        
        if current_interactions_pd.empty:
            print(f"No interactions in 'log' for round {round_to_process}. Skipping sequence curation and training for this round.")
            return 

        if 'relevance' in current_interactions_pd.columns:
            current_interactions_pd['relevance'] = current_interactions_pd['relevance'].apply(lambda x: 1 if x > 0 else 0)

        for _, row in current_interactions_pd.iterrows():
            user_idx = row[self.user_id_col]
            item_id = row[self.item_id_col]
            relevance = row['relevance']
            price = self.item_prices.get(item_id, 0.0) # Get price

            # Use the round_to_process as the timestamp for these interactions
            interaction_element = (round_to_process, item_id, price, relevance) 
            
            # Append to existing sequence
            if user_idx not in self.user_interaction_sequences:
                self.user_interaction_sequences[user_idx] = []
            self.user_interaction_sequences[user_idx].append(interaction_element)
            
            # Keep sequence length limited
            if len(self.user_interaction_sequences[user_idx]) > self.max_sequence_length:
                self.user_interaction_sequences[user_idx] = self.user_interaction_sequences[user_idx][-self.max_sequence_length:]
        
        # Prepare data for model training (batching, padding, masking)
        trainable_users = [u for u, seq in self.user_interaction_sequences.items() if len(seq) >= 2] 

        if not trainable_users:
            print(f"No users with sufficient history (>=2 interactions) to train the model in round {round_to_process}.")
            return

        for epoch in range(self.epochs):
            total_loss = 0
            num_batches = 0

            np.random.shuffle(trainable_users) # Shuffle users for each epoch

            for i in range(0, len(trainable_users), self.batch_size):
                batch_user_ids = trainable_users[i:i + self.batch_size]
                
                batch_item_seqs = []
                batch_masks = []
                batch_targets = []

                for user_id in batch_user_ids:
                    sequence = self.user_interaction_sequences[user_id]
                    sequence.sort(key=lambda x: x[0]) 

                    item_ids_raw = [s[1] for s in sequence]
                    
                    # Ensure input and target sequences are not empty due to previous filtering
                    if not item_ids_raw:
                        continue

                    # Input sequence for the model
                    input_seq = item_ids_raw
                    target_seq = item_ids_raw[1:] + [self.padding_item_idx] 

                    # Padding
                    current_seq_len = len(input_seq)
                    padding_len = self.max_sequence_length - current_seq_len

                    padded_input_seq = input_seq + [self.padding_item_idx] * padding_len
                    padded_target_seq = target_seq + [self.padding_item_idx] * padding_len
                    
                    # Mask: True for padding, False for real items
                    current_mask = [False] * current_seq_len + [True] * padding_len

                    # Ensure lengths are capped
                    padded_input_seq = padded_input_seq[:self.max_sequence_length]
                    padded_target_seq = padded_target_seq[:self.max_sequence_length]
                    current_mask = current_mask[:self.max_sequence_length]

                    batch_item_seqs.append(padded_input_seq)
                    batch_masks.append(current_mask)
                    batch_targets.append(padded_target_seq)

                # Skip batch if it ended up empty
                if not batch_item_seqs:
                    continue

                # Convert to tensors
                item_seqs_tensor = torch.tensor(batch_item_seqs, dtype=torch.long).to(self.device)
                masks_tensor = torch.tensor(batch_masks, dtype=torch.bool).to(self.device)
                targets_tensor = torch.tensor(batch_targets, dtype=torch.long).to(self.device)

                # Forward pass
                logits = self.model(item_seqs_tensor, masks_tensor)

                logits_for_loss = logits.view(-1, self.total_num_items)
                targets_for_loss = targets_tensor.view(-1)

                loss = self.criterion(logits_for_loss, targets_for_loss)

                # Backward pass and optimize
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            print(f"Epoch {epoch+1} finished. Average Loss: {avg_loss:.4f}")

    def predict(self, log: DataFrame, k: int, users: DataFrame, items: DataFrame, user_features=None, item_features=None, filter_seen_items: bool = True):
        """
        Generate recommendations for users.
        """
        self.model.eval() # Set model to evaluation mode
        
        # Get all item prices
        item_prices_pd = item_features.select(self.item_id_col, "price").toPandas()
        
        # Create a mapping from item_id to price
        item_id_to_price_dict = item_prices_pd.set_index(self.item_id_col)['price'].to_dict()
        
        # Create a full price tensor, aligned by item_id
        full_item_prices = [item_id_to_price_dict.get(i, 0.0) for i in range(self.total_num_items)]
        item_prices_tensor = torch.tensor(full_item_prices, dtype=torch.float, device=self.device)

        # Get all unique users from the input 'users_df"
        users_to_predict_pd = users.select(self.user_id_col).distinct().toPandas()
        all_user_ids = users_to_predict_pd[self.user_id_col].tolist()

        # Collect seen items for filtering
        seen_items = defaultdict(set)
        if filter_seen_items:
            log_pd = log.toPandas()
            for _, row in log_pd.iterrows():
                seen_items[row[self.user_id_col]].add(row[self.item_id_col])

        # Prepare batch input for the Transformer
        batch_item_seqs = []
        batch_masks = []
        actual_user_ids_in_batch = []

        # Iterate through users and prepare their sequences
        for user_id in all_user_ids:
            sequence = self.user_interaction_sequences.get(user_id, [])

            if not sequence:
                continue

            sequence.sort(key=lambda x: x[0])
            item_ids_raw = [s[1] for s in sequence][-self.max_sequence_length:] 

            current_seq_len = len(item_ids_raw)
            padding_len = self.max_sequence_length - current_seq_len
            
            padded_input_seq = item_ids_raw + [self.padding_item_idx] * padding_len 
            mask = [False] * current_seq_len + [True] * padding_len 

            batch_item_seqs.append(padded_input_seq)
            batch_masks.append(mask)
            actual_user_ids_in_batch.append(user_id)

        if not actual_user_ids_in_batch:
            current_spark_session = SparkSession.builder.getOrCreate()
            return pandas_to_spark(pd.DataFrame(columns=[self.user_id_col, self.item_id_col, "relevance"]), current_spark_session)

        item_seqs_tensor = torch.tensor(batch_item_seqs, dtype=torch.long).to(self.device)
        masks_tensor = torch.tensor(batch_masks, dtype=torch.bool).to(self.device)

        recommendations_list = []

        with torch.no_grad():
            model_output_logits = self.model(item_seqs_tensor, masks_tensor)

            sequence_lengths = (~masks_tensor).sum(dim=1)
            last_item_indices = torch.clamp(sequence_lengths - 1, min=0)
            
            user_logits_batch = model_output_logits[torch.arange(model_output_logits.size(0), device=model_output_logits.device), last_item_indices]

            global_candidate_mask = torch.ones(self.total_num_items, dtype=torch.bool, device=self.device)
            if self.padding_item_idx < self.total_num_items:
                global_candidate_mask[self.padding_item_idx] = False

            for i, user_id in enumerate(actual_user_ids_in_batch):
                current_user_logits = user_logits_batch[i]

                # Initialize a per-user candidate mask
                user_candidate_mask = global_candidate_mask.clone()
                
                # Filter out items seen by this user
                if filter_seen_items:
                    current_seen_items = seen_items.get(user_id, set())
                    for seen_item_id in current_seen_items:
                        if seen_item_id < self.total_num_items:
                            user_candidate_mask[seen_item_id] = False 

                # Apply the candidate mask
                current_user_logits_masked = current_user_logits.masked_fill(~user_candidate_mask, float('-inf'))

                # Convert logits to probabilities
                pseudo_probabilities = torch.sigmoid(current_user_logits_masked) # This will correctly set -inf values to 0

                # Calculate Expected Revenue for all items for the current user
                expected_revenue_scores = pseudo_probabilities * item_prices_tensor 
                
                # Get top-k based on Expected Revenue Scores
                top_k_values_rev, top_k_indices_rev = torch.topk(expected_revenue_scores, k=k, dim=-1)

                for item_rank in range(k):
                    item_id = top_k_indices_rev[item_rank].item()
                    original_logit_for_selected_item = current_user_logits[item_id].item()
                    relevance_score_for_output = torch.sigmoid(torch.tensor(original_logit_for_selected_item)).item()
                    
                    recommendations_list.append({
                        self.user_id_col: user_id,
                        self.item_id_col: item_id,
                        "relevance": relevance_score_for_output
                    })

        recommendations_pd = pd.DataFrame(recommendations_list)
        if recommendations_pd.empty:
            current_spark_session = SparkSession.builder.getOrCreate()
            return pandas_to_spark(pd.DataFrame(columns=[self.user_id_col, self.item_id_col, "relevance"]), current_spark_session)
            
        current_spark_session = SparkSession.builder.getOrCreate()
        return pandas_to_spark(recommendations_pd, current_spark_session)


    def _update_item_prices(self, items_df: DataFrame):
        """
        Updates the internal item prices dictionary from the items DataFrame.
        """
        for row in items_df.toPandas().itertuples():
            self.item_prices[row.item_idx] = row.price

    def _prepare_item_features_for_embedding(self, item_features_df: DataFrame) -> DataFrame:
        """
        Processes item features and prepares them for use as embeddings.
        """
        # Dynamically identify categorical and numerical columns starting with item_attr_
        item_categorical_cols = [
            col.name for col in item_features_df.schema
            if col.name.startswith("item_attr_") and isinstance(col.dataType, StringType)
        ]
        item_numerical_cols = [
            col.name for col in item_features_df.schema
            if col.name.startswith("item_attr_") and isinstance(col.dataType, (DoubleType, IntegerType, LongType))
        ]
        
        # Process Categorical Features
        processed_item_features = process_categorical_features(
            item_features_df,
            categorical_cols=item_categorical_cols,
            prefix="item_attr_"
        )

        # Process Numerical Features (scaling)
        final_processed_item_features = scale_numerical_features(
            processed_item_features,
            numerical_cols=item_numerical_cols,
            output_prefix="item_attr_"
        )

        # Assemble all processed features into a single vector
        item_vector_input_cols = []
        for col in item_categorical_cols:
            item_vector_input_cols.append(f"item_attr_{col}_encoded")
        if item_numerical_cols:
             item_vector_input_cols.append("item_attr_numerical_features_scaled")
        
        # Handle case where no item attributes are found
        if not item_vector_input_cols:
            print("Warning: No item_attr_ columns found for feature vector creation. Item embeddings will be learned from scratch.")
            return item_features_df.select("item_idx")

        assembler = VectorAssembler(
            inputCols=item_vector_input_cols,
            outputCol="features_vec"
        )
        items_with_feature_vectors = assembler.transform(final_processed_item_features)

        # Select only item_idx and the new features_vec
        return items_with_feature_vectors.select("item_idx", "features_vec")

def process_categorical_features(df: DataFrame, categorical_cols: list, prefix: str = "") -> DataFrame:
    """
    Applies StringIndexer and OneHotEncoder to categorical columns.
    Converts string categories into numerical indices and then into one-hot vectors.
    """
    for col_name in categorical_cols:
        indexed_col = f"{prefix}{col_name}_indexed"
        encoded_col = f"{prefix}{col_name}_encoded"

        # Converts string categories into numerical indices
        indexer = StringIndexer(
            inputCol=col_name,
            outputCol=indexed_col,
            handleInvalid="keep"
        )
        df = indexer.fit(df).transform(df)

        # OneHotEncoder
        encoder = OneHotEncoder(
            inputCol=indexed_col,
            outputCol=encoded_col,
            dropLast=False
        )
        df = encoder.fit(df).transform(df)

        # Drop the intermediate indexed column
        df = df.drop(indexed_col)
    return df

def scale_numerical_features(df: DataFrame, numerical_cols: list, output_prefix: str = "") -> DataFrame:
    """
    Scales numerical features using StandardScaler.
    Assembles numerical features into a vector, scales them, and then drops the raw vector.
    """
    if not numerical_cols:
        return df

    # Assemble numerical features into a single vector column
    assembler = VectorAssembler(inputCols=numerical_cols, outputCol="numerical_features_raw")
    df_assembled = assembler.transform(df)

    # Apply StandardScaler
    scaler = StandardScaler(
        inputCol="numerical_features_raw",
        outputCol=f"{output_prefix}numerical_features_scaled",
        withStd=True,
        withMean=True
    )
    scaler_model = scaler.fit(df_assembled)
    df_scaled = scaler_model.transform(df_assembled)

    # Drop the intermediate raw numerical features vector
    df_scaled = df_scaled.drop("numerical_features_raw")

    return df_scaled