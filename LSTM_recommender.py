import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import shutil

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
from sim4rec.utils import  pandas_to_spark

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

# Import competition modules
from data_generator import CompetitionDataGenerator
from simulator import CompetitionSimulator
from sample_recommenders import (
    RandomRecommender,
    PopularityRecommender,
    ContentBasedRecommender, 
    SVMRecommender, 
)
from config import DEFAULT_CONFIG, EVALUATION_METRICS

# Cell: Define custom recommender template
"""
## MyRecommender Template
Below is a template class for implementing a custom recommender system.
Students should extend this class with their own recommendation algorithm.
"""

#####################################################################################################
#DEFINES THE ACTUAL NN ARCHITECTURE FOR LSTM MODEL

class LSTMRecModel(nn.Module):
    """
    LSTM-based sequential recommendation model.
    Predicts the next item in a sequence using LSTM layers.
    """
    def __init__(self, num_items: int, embedding_dim: int, hidden_size: int,
                 num_layers: int, dropout_rate: float, padding_idx: int,
                 use_price_embedding: bool = True, price_embedding_dim: int = 16):
        super().__init__()
        self.num_items = num_items # Total number of unique items in the dataset
        self.embedding_dim = embedding_dim # Dimensionality of item embeddings
        self.hidden_size = hidden_size # Number of features in the LSTM hidden state
        self.num_layers = num_layers # Number of recurrent layers in the LSTM
        self.padding_idx = padding_idx # Index used for padding items in sequences
        self.use_price_embedding = use_price_embedding # Whether to incorporate price information
        
        # Item embedding layer: maps each item ID to a dense vector
        # padding_idx makes embeddings for this index 0 and doesn't update them during training
        self.item_embedding = nn.Embedding(num_items, embedding_dim, padding_idx=padding_idx)
        
        # Price embedding (optional)
        if use_price_embedding:
            # Discretize prices into bins for embedding
            self.price_bins = 20  # Number of price bins (e.g., for prices 0-1000, 20 bins means 50 units per bin)
            # Price embedding layer: maps each price bin to a dense vector
            self.price_embedding = nn.Embedding(self.price_bins, price_embedding_dim)
            # The input size to the LSTM will be the concatenation of item and price embeddings
            lstm_input_size = embedding_dim + price_embedding_dim
        else:
            # If no price embedding, LSTM input size is just the item embedding dimension
            lstm_input_size = embedding_dim
            
        # LSTM layers: processes sequential input
        self.lstm = nn.LSTM(
            input_size=lstm_input_size, # Dimension of the input features
            hidden_size=hidden_size,    # Dimension of the hidden state
            num_layers=num_layers,      # Number of stacked LSTM layers
            # Dropout applied to the output of each LSTM layer except the last
            dropout=dropout_rate if num_layers > 1 else 0,
            batch_first=True            # Input and output tensors are (batch, seq_len, features)
        )
        
        # Dropout layer: applied to the combined input embeddings and LSTM output
        self.dropout = nn.Dropout(dropout_rate)
        
        # Output projection layer: maps LSTM's hidden state to item scores (logits)
        self.output_projection = nn.Linear(hidden_size, num_items)
        
        # Loss function: CrossEntropyLoss is suitable for multi-class classification (predicting the next item)
        # ignore_index ensures that padding tokens in the target sequence do not contribute to the loss
        self.criterion = nn.CrossEntropyLoss(ignore_index=padding_idx)
        
    def _discretize_prices(self, prices: torch.Tensor) -> torch.Tensor:
        """
        Convert continuous prices to discrete bins for embedding.
        This helps in handling price as a categorical-like feature.
        """
        # Clamp prices to a predefined range to prevent outliers from distorting binning
        prices_clamped = torch.clamp(prices, 0, 1000) # Assuming max price of 1000, adjust as needed
        # Discretize prices: e.g., if price_bins=20, each bin covers 50 units (1000/20)
        price_bins = (prices_clamped / 50).long() # Convert to long for embedding lookup
        # Ensure bin indices are within the valid range [0, self.price_bins - 1]
        price_bins = torch.clamp(price_bins, 0, self.price_bins - 1)
        return price_bins
        
    def forward(self, item_seq: torch.Tensor, price_seq: torch.Tensor = None,
                mask: torch.Tensor = None, hidden_state: tuple = None):
        """
        Forward pass through the LSTM model.
        
        Args:
            item_seq: (batch_size, seq_len) - sequence of item IDs for each user in the batch.
            price_seq: (batch_size, seq_len) - sequence of prices corresponding to item_seq (optional).
            mask: (batch_size, seq_len) - padding mask (True for padding, False for real items). Not directly
                  used in the LSTM layer itself, but useful for other components or for future additions.
            hidden_state: tuple of (h_0, c_0) for LSTM initialization, allowing for stateful processing
                          or passing initial states (e.g., from user features).
            
        Returns:
            logits: (batch_size, seq_len, num_items) - raw prediction scores for each item at each timestep.
            final_hidden: tuple of final hidden states (h_n, c_n) from the LSTM.
        """
        batch_size, seq_len = item_seq.size()
        
        # Get item embeddings
        item_embs = self.item_embedding(item_seq) # Output: (batch_size, seq_len, embedding_dim)
        
        # Combine with price embeddings if available
        if self.use_price_embedding and price_seq is not None:
            price_bins = self._discretize_prices(price_seq) # Convert continuous prices to discrete bins
            price_embs = self.price_embedding(price_bins)   # Output: (batch_size, seq_len, price_embedding_dim)
            # Concatenate item and price embeddings along the feature dimension
            lstm_input = torch.cat([item_embs, price_embs], dim=-1)
        else:
            lstm_input = item_embs
            
        # Apply dropout to the combined input embeddings
        lstm_input = self.dropout(lstm_input)
        
        # Pass through LSTM layers
        if hidden_state is not None:
            # If initial hidden state is provided, use it
            lstm_output, final_hidden = self.lstm(lstm_input, hidden_state)
        else:
            # Otherwise, LSTM initializes hidden state to zeros
            lstm_output, final_hidden = self.lstm(lstm_input)
            
        # Apply dropout to the LSTM output
        lstm_output = self.dropout(lstm_output)
        
        # Project the LSTM output (hidden states) to prediction scores for all items
        logits = self.output_projection(lstm_output) # Output: (batch_size, seq_len, num_items)
        
        return logits, final_hidden
    
# manages the overall workflow, including data handling, training loops, and prediction
class LSTMRecommender:
    """
    LSTM-based Sequential Recommender System
    Manages data preprocessing, model training, and recommendation generation.
    """
    def __init__(self, seed=None, max_sequence_length=100,
                 item_embedding_size=64, hidden_size=128, num_layers=2,
                 dropout_rate=0.3, use_price_embedding=True, price_embedding_dim=16,
                 learning_rate=0.001, epochs=10, batch_size=32,
                 user_id_col: str = "user_idx",
                 item_id_col: str = "item_idx"):
        
        # Set random seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            
        self.max_sequence_length = max_sequence_length # Maximum length of item sequences
        # Stores historical interactions for each user: {user_id: [(timestamp, item_id, price, relevance), ...]}
        self.user_interaction_sequences = defaultdict(list)
        self.item_prices = {} # Dictionary to store current item prices: {item_id: price}
        self.user_id_col = user_id_col # Column name for user ID
        self.item_id_col = item_id_col # Column name for item ID
        
        
        
        # Determine device for PyTorch operations (GPU if available, else CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model hyperparameters (passed to LSTMRecModel)
        self.item_embedding_size = item_embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.use_price_embedding = use_price_embedding
        self.price_embedding_dim = price_embedding_dim
        # Training hyperparameters
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        
        self._current_round = 0       # Tracks the current training round (useful for timestamping interactions)
        self._model_initialized = False # Flag to ensure model is initialized only once
        
        print(f"LSTM Recommender initialized with:")
        print(f"  - Hidden size: {hidden_size}")
        print(f"  - Number of layers: {num_layers}")
        print(f"  - Dropout rate: {dropout_rate}")
        print(f"  - Use price embedding: {use_price_embedding}")
        print(f"  - Device: {self.device}")

    def _create_training_examples(self, sequence):
        """
        Create multiple training examples from a single user sequence using sliding window approach.
        This significantly increases the amount of training data available.
        
        Args:
            sequence: List of (timestamp, item_id, price, relevance) tuples
            
        Returns:
            List of (input_items, input_prices, target_item) tuples
        """
        if not sequence: # Handle empty sequence directly
            return []
            
        # Sort by timestamp to ensure correct order
        sequence.sort(key=lambda x: x[0])
        
        training_examples = []
        
        # New logic: If sequence has at least one item, create an example where input is empty, target is the first item
        # This helps with cold-start users or predicting their first interaction
        if len(sequence) >= 1:
            first_item = sequence[0][1]
            # Input items and prices are empty, target is the first item
            training_examples.append(([], [], first_item))

        # Original logic: Create training examples using all possible prefixes for subsequent interactions
        # For sequence [A, B, C, D], create:
        # Input: [A], Target: B
        # Input: [A, B], Target: C
        # Input: [A, B, C], Target: D
        
        for i in range(1, len(sequence)):  # Start from 1 as 'i' is the index of the target item
            input_items = [s[1] for s in sequence[:i]]  # Items up to position i-1
            input_prices = [s[2] for s in sequence[:i]]  # Prices up to position i-1
            target_item = sequence[i][1]  # Next item to predict (item at position i)
            
            # Limit input sequence length
            if len(input_items) > self.max_sequence_length:
                input_items = input_items[-self.max_sequence_length:]
                input_prices = input_prices[-self.max_sequence_length:]
            
            training_examples.append((input_items, input_prices, target_item))
        
        return training_examples

    def fit(self, log: DataFrame = None, user_features: DataFrame = None,
            item_features: DataFrame = None):
        """
        Train the LSTM recommender model based on interaction history.
        This method is designed to be called iteratively with new data in a streaming/batch setup.
        
        Args:
            log: Spark DataFrame containing new interactions (user_idx, item_idx, relevance).
            user_features: Spark DataFrame with user features (optional, not directly used by LSTM for now).
            item_features: Spark DataFrame with item features (used to update item prices).
        """
        round_to_process = self._current_round # Capture current round number
        self._current_round += 1              # Increment for the next call
        
        print(f"\n--- LSTM Training Round {round_to_process} ---")
        
        # Lazily initialize model and optimizer on the first fit call
        if not self._model_initialized:
            # Determine total number of items including a dedicated padding index
            max_actual_item_id = item_features.select(sf.max(self.item_id_col)).collect()[0][0]
            num_real_items = max_actual_item_id + 1 # Assuming item_idx starts from 0
            self.padding_item_idx = num_real_items # Assign the next available integer as padding index
            self.total_num_items = num_real_items + 1 # Total unique items + 1 for padding
            
            # Initialize the PyTorch LSTM model
            self.model = LSTMRecModel(
                num_items=self.total_num_items,
                embedding_dim=self.item_embedding_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout_rate=self.dropout_rate,
                padding_idx=self.padding_item_idx,
                use_price_embedding=self.use_price_embedding,
                price_embedding_dim=self.price_embedding_dim
            ).to(self.device) # Move model to appropriate device (CPU/GPU)
            
            # Use AdamW optimizer with weight decay for better regularization
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
            
            # Learning rate scheduler for adaptive learning rate
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', patience=3, factor=0.5
            )
            
            self._model_initialized = True # Set flag to true
            
            print(f"LSTM Model initialized with {self.total_num_items} items, padding_idx={self.padding_item_idx}")
        
        # Set model to training mode (enables dropout, batch norm etc.)
        self.model.train()
        
        # Update internal item prices dictionary from the provided item_features
        self._update_item_prices(item_features)
        
        # Process new interactions from the current log DataFrame
        current_interactions_pd = log.toPandas()
        
        if current_interactions_pd.empty:
            print(f"No interactions in log for round {round_to_process}. Skipping training.")
            return # Exit if no new interactions
            
        # Ensure 'relevance' column is binary (0 or 1)
        if 'relevance' in current_interactions_pd.columns:
            current_interactions_pd['relevance'] = current_interactions_pd['relevance'].apply(
                lambda x: 1 if x > 0 else 0
            )
            
        print(f"Processing {len(current_interactions_pd)} new interactions for round {round_to_process}")
        
        # Update user interaction sequences with new data
        for _, row in current_interactions_pd.iterrows():
            user_idx = row[self.user_id_col]
            item_id = row[self.item_id_col]
            relevance = row['relevance']
            price = self.item_prices.get(item_id, 0.0) # Get price, default to 0.0 if not found
            
            # Store interaction with timestamp (round_to_process), item ID, price, and relevance
            interaction_element = (round_to_process, item_id, price, relevance)
            
            # Append to existing sequence or create a new one
            if user_idx not in self.user_interaction_sequences:
                self.user_interaction_sequences[user_idx] = []
            self.user_interaction_sequences[user_idx].append(interaction_element)
            
            # Keep sequence length limited to max_sequence_length
            if len(self.user_interaction_sequences[user_idx]) > self.max_sequence_length:
                self.user_interaction_sequences[user_idx] = self.user_interaction_sequences[user_idx][-self.max_sequence_length:]
        
        print(f"Sequences updated. Total users with interactions: {len(self.user_interaction_sequences)}")
        
        # Lower threshold for trainable users and create more training examples
        trainable_users = [u for u, seq in self.user_interaction_sequences.items() if len(seq) >= 1]
        
        if not trainable_users:
            print(f"No users with sufficient history (>=1 interactions) for training.")
            return # Exit if no users to train on
        
        # Create all training examples from all user sequences
        all_training_examples = []
        for user_id in trainable_users:
            sequence = self.user_interaction_sequences[user_id]
            user_examples = self._create_training_examples(sequence)
            all_training_examples.extend(user_examples)
        
        if not all_training_examples:
            print("No training examples generated.")
            return
            
        print(f"Generated {len(all_training_examples)} training examples from {len(trainable_users)} users")
        print(f"Starting LSTM training for {self.epochs} epochs")
        
        # Training loop
        for epoch in range(self.epochs):
            total_loss = 0
            num_batches = 0
            
            # Shuffle training examples each epoch
            np.random.shuffle(all_training_examples)
            
            # Iterate through training examples in mini-batches
            for i in tqdm.tqdm(range(0, len(all_training_examples), self.batch_size),
                               desc=f"Epoch {epoch+1}/{self.epochs}"):
                batch_examples = all_training_examples[i:i + self.batch_size]
                
                batch_item_seqs = []
                batch_price_seqs = []
                batch_targets = []
                
                for input_items, input_prices, target_item in batch_examples:
                    if not input_items:  # Skip empty sequences
                        continue
                        
                    # Padding logic
                    current_seq_len = len(input_items)
                    padding_len = self.max_sequence_length - current_seq_len
                    
                    if padding_len > 0:
                        # Pad inputs to max_sequence_length
                        padded_input_items = input_items + [self.padding_item_idx] * padding_len
                        padded_input_prices = input_prices + [0.0] * padding_len # Pad prices with 0
                    else:
                        # Truncate if sequence is longer than max_sequence_length
                        padded_input_items = input_items[-self.max_sequence_length:]
                        padded_input_prices = input_prices[-self.max_sequence_length:]
                        
                    batch_item_seqs.append(padded_input_items)
                    batch_price_seqs.append(padded_input_prices)
                    batch_targets.append(target_item)
                
                if not batch_item_seqs: # Skip if the batch ended up empty after filtering
                    continue
                
                # Convert lists to PyTorch tensors
                item_seqs_tensor = torch.tensor(batch_item_seqs, dtype=torch.long).to(self.device)
                price_seqs_tensor = torch.tensor(batch_price_seqs, dtype=torch.float).to(self.device)
                targets_tensor = torch.tensor(batch_targets, dtype=torch.long).to(self.device)
                
                # Forward pass: get logits from the model
                logits, _ = self.model(item_seqs_tensor, price_seqs_tensor)
                
                # IMPROVED: Use the last valid position for each sequence to get predictions
                # Find the actual sequence lengths (before padding)
                sequence_lengths = []
                for seq in batch_item_seqs:
                    actual_length = 0
                    for item in seq:
                        if item != self.padding_item_idx:
                            actual_length += 1
                        else:
                            break
                    sequence_lengths.append(max(actual_length - 1, 0))  # -1 because we want the last valid position
                
                # Extract logits at the last valid position for each sequence
                batch_indices = torch.arange(logits.size(0)).to(self.device)
                sequence_lengths_tensor = torch.tensor(sequence_lengths, dtype=torch.long).to(self.device)
                
                # Get logits at the last valid timestep for each sequence
                logits_for_loss = logits[batch_indices, sequence_lengths_tensor]  # (batch_size, num_items)
                
                # Calculate loss
                loss = self.model.criterion(logits_for_loss, targets_tensor)
                
                # Backward pass and optimization
                self.optimizer.zero_grad() # Clear previous gradients
                loss.backward()             # Compute gradients
                # Clip gradients to prevent exploding gradients, common in RNNs
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()       # Update model weights
                
                total_loss += loss.item() # Accumulate loss
                num_batches += 1
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            print(f"Epoch {epoch+1} finished. Average Loss: {avg_loss:.4f}")
            
            # Update learning rate based on loss
            self.scheduler.step(avg_loss)

    def predict(self, log: DataFrame, k: int, users: DataFrame, items: DataFrame,
                user_features=None, item_features=None, filter_seen_items: bool = True):
        """
        Generate recommendations using the trained LSTM model.
        
        Args:
            log: Spark DataFrame with recent interactions (used for filtering seen items).
            k: Number of top items to recommend for each user.
            users: Spark DataFrame with users for whom to generate recommendations.
            items: Spark DataFrame with all available items.
            user_features: User features (optional, not used in current implementation).
            item_features: Item features (optional, not used in current implementation beyond prices).
            filter_seen_items: Boolean, if True, filters out items a user has already interacted with.
            
        Returns:
            DataFrame: Spark DataFrame with recommendations (user_idx, item_idx, relevance).
        """
        # Set model to evaluation mode (disables dropout, etc.)
        self.model.eval()
        # Ensure item prices are up-to-date for prediction (crucial for revenue-based sorting)
        self._update_item_prices(items) #  called before prediction if item_features is available

        # Get all unique user IDs for whom to generate recommendations
        users_to_predict_pd = users.select(self.user_id_col).distinct().toPandas()
        all_user_ids = users_to_predict_pd[self.user_id_col].tolist()
        
        # Collect seen items for efficient filtering during prediction
        seen_items = defaultdict(set)
        if filter_seen_items:
            log_pd = log.toPandas()
            for _, row in log_pd.iterrows():
                seen_items[row[self.user_id_col]].add(row[self.item_id_col])
        
        recommendations_list = [] # List to store generated recommendations

        # Prepare item prices as a PyTorch tensor for efficient calculation
        # Create a tensor of prices for all items, matching the dimension of model logits
        all_item_ids = list(range(self.total_num_items))
        # Ensure prices are available for all items the model might predict
        # Default to 0.0 or a small value for items without explicit prices
        prices_for_tensor = [self.item_prices.get(i, 0.0) for i in all_item_ids]
        item_prices_tensor = torch.tensor(prices_for_tensor, dtype=torch.float, device=self.device)
        # Apply log1p transformation to prices once for all items
        price_factors_tensor = torch.log1p(item_prices_tensor)
        
        # Process users in batches for prediction
        for i in tqdm.tqdm(range(0, len(all_user_ids), self.batch_size), desc="Generating recommendations"):
            batch_user_ids = all_user_ids[i:i + self.batch_size]
            
            batch_item_seqs = []
            batch_price_seqs = []
            batch_masks = [] # Mask is primarily used to determine actual sequence length for `last_item_indices`
            actual_user_ids_in_batch = [] # Store user IDs corresponding to the batch tensors
            
            for user_id in batch_user_ids:
                sequence = self.user_interaction_sequences.get(user_id, [])
                
                if not sequence:
                    # Users with no history cannot be processed by a sequence model
                    continue
                
                # Sort by timestamp and get the most recent interactions up to max_sequence_length
                sequence.sort(key=lambda x: x[0])
                recent_sequence = sequence[-self.max_sequence_length:]
                
                item_ids = [s[1] for s in recent_sequence]
                prices = [s[2] for s in recent_sequence]
                
                # Padding logic for inference input
                current_seq_len = len(item_ids)
                padding_len = self.max_sequence_length - current_seq_len
                
                if padding_len > 0:
                    padded_items = item_ids + [self.padding_item_idx] * padding_len
                    padded_prices = prices + [0.0] * padding_len
                    mask = [False] * current_seq_len + [True] * padding_len
                else:
                    padded_items = item_ids[:self.max_sequence_length]
                    padded_prices = prices[:self.max_sequence_length]
                    mask = [False] * self.max_sequence_length
                    
                batch_item_seqs.append(padded_items)
                batch_price_seqs.append(padded_prices)
                batch_masks.append(mask)
                actual_user_ids_in_batch.append(user_id)
            
            if not actual_user_ids_in_batch: # Skip if the batch became empty
                continue
            
            # Convert to PyTorch tensors and move to device
            item_seqs_tensor = torch.tensor(batch_item_seqs, dtype=torch.long).to(self.device)
            price_seqs_tensor = torch.tensor(batch_price_seqs, dtype=torch.float).to(self.device)
            masks_tensor = torch.tensor(batch_masks, dtype=torch.bool).to(self.device)
            
            with torch.no_grad(): # Disable gradient calculations during inference
                # Get model predictions
                # logits: (batch_size, seq_len, num_items)
                logits, _ = self.model(item_seqs_tensor, price_seqs_tensor, masks_tensor)
                
                # Identify the length of each actual sequence (non-padded part)
                sequence_lengths = (~masks_tensor).sum(dim=1) # Counts False (non-padded) elements
                # Get the index of the last valid item in each sequence
                last_item_indices = torch.clamp(sequence_lengths - 1, min=0) # Handles empty sequences (clamped to 0)
                
                # Extract logits for the prediction *after* the last valid item for each user
                # This corresponds to the next item prediction
                user_logits_batch = logits[torch.arange(logits.size(0)), last_item_indices]

                # Convert logits to probabilities (sigmoid)
                probabilities = torch.sigmoid(user_logits_batch) # (batch_size, num_items)

                # Multiply probabilities by price factors to get revenue-aware scores
                revenue_aware_scores = probabilities * price_factors_tensor 

                # Generate a dynamic candidate mask for the batch
                batch_candidate_mask = torch.ones_like(revenue_aware_scores, dtype=torch.bool, device=self.device)
                batch_candidate_mask[:, self.padding_item_idx] = False

                # Filter out items already seen by each user in the batch
                if filter_seen_items:
                    for j, user_id in enumerate(actual_user_ids_in_batch):
                        user_seen_items = seen_items.get(user_id, set())
                        for seen_item in user_seen_items:
                            if seen_item < self.total_num_items:
                                batch_candidate_mask[j, seen_item] = False # Exclude seen items for this user
                
                # Apply the candidate mask to revenue-aware scores
                # Set scores of invalid/seen items to -infinity so they are not selected by topk
                filtered_revenue_scores = revenue_aware_scores.masked_fill(~batch_candidate_mask, -torch.inf)
                top_k_values, top_k_indices = torch.topk(filtered_revenue_scores, k=k, dim=-1)
              
                
                # Generate recommendations for each user in the batch
                for j, user_id in enumerate(actual_user_ids_in_batch):
                    user_top_k_values = top_k_values[j]
                    user_top_k_indices = top_k_indices[j]

                    for rank in range(k):
                        recommended_item_id = user_top_k_indices[rank].item()
                        final_relevance_score = user_top_k_values[rank].item() # This is already the revenue-aware score

                        recommendations_list.append({
                            self.user_id_col: user_id,
                            self.item_id_col: recommended_item_id,
                            "relevance": final_relevance_score
                        })

        
        # Convert the list of recommendations to a Pandas DataFrame, then to Spark DataFrame
        recommendations_pd = pd.DataFrame(recommendations_list)
        if recommendations_pd.empty:
            current_spark_session = SparkSession.builder.getOrCreate()
            empty_df = pd.DataFrame(columns=[self.user_id_col, self.item_id_col, "relevance"])
            return pandas_to_spark(empty_df, current_spark_session) 
            
        current_spark_session = SparkSession.builder.getOrCreate()
        return pandas_to_spark(recommendations_pd, current_spark_session) 

    def _update_item_prices(self, items_df: DataFrame):
        """
        Update internal item prices dictionary from the items DataFrame.
        This ensures the model has access to the most recent item prices for embedding.
        """
        for row in items_df.toPandas().itertuples():
            # Assumes 'price' column exists in the input items_df
            if 'price' in items_df.columns:
                self.item_prices[row.item_idx] = row.price
            else:
                # Default to 0.0 if no price column is found
                self.item_prices[row.item_idx] = 0.0

   

    def get_user_sequence_stats(self):
        """
        Get statistics about user interaction sequences stored in the recommender.
        Useful for monitoring data state.
        """
        if not self.user_interaction_sequences:
            return {"total_users": 0, "avg_sequence_length": 0, "max_sequence_length": 0, "min_sequence_length": 0}
            
        sequence_lengths = [len(seq) for seq in self.user_interaction_sequences.values()]
        return {
            "total_users": len(self.user_interaction_sequences),
            "avg_sequence_length": np.mean(sequence_lengths),
            "max_sequence_length": max(sequence_lengths),
            "min_sequence_length": min(sequence_lengths)
        }
    
    def _prepare_item_features_for_embedding(self, item_features_df: DataFrame) -> DataFrame:
        """
        Processes item features and prepares them for use as embeddings.
        This involves one-hot encoding categorical features and scaling numerical ones,
        then assembling them into a single vector.

        Args:
            item_features_df: DataFrame containing item attributes.

        Returns:
            DataFrame: Item features with a 'features_vec' column suitable for embedding.
        """
        # Dynamically identify categorical and numerical columns starting with "item_attr_"
        item_categorical_cols = [
            col.name for col in item_features_df.schema
            if col.name.startswith("item_attr_") and isinstance(col.dataType, StringType)
        ]
        item_numerical_cols = [
            col.name for col in item_features_df.schema
            if col.name.startswith("item_attr_") and isinstance(col.dataType, (DoubleType, IntegerType, LongType))
        ]
        
        # 1. Process Categorical Features
        processed_item_features = process_categorical_features(
            item_features_df,
            categorical_cols=item_categorical_cols,
            prefix="item_attr_"
        )

        # 2. Process Numerical Features (scaling)
        final_processed_item_features = scale_numerical_features(
            processed_item_features,
            numerical_cols=item_numerical_cols,
            output_prefix="item_attr_"
        )

        # 3. Assemble all processed features into a single vector
        item_vector_input_cols = []
        for col in item_categorical_cols:
            item_vector_input_cols.append(f"item_attr_{col}_encoded")
        if item_numerical_cols:
             item_vector_input_cols.append("item_attr_numerical_features_scaled")
        
        # Handle case where no item attributes are found
        if not item_vector_input_cols:
            print("Warning: No item_attr_ columns found for feature vector creation. Item embeddings will be learned from scratch.")
            # If no features, just return item_idx. The embedding layer will learn from ID.
            return item_features_df.select("item_idx")

        assembler = VectorAssembler(
            inputCols=item_vector_input_cols,
            outputCol="features_vec" # Output column for the combined feature vector
        )
        items_with_feature_vectors = assembler.transform(final_processed_item_features)

        # Select only item_idx and the new features_vec
        return items_with_feature_vectors.select("item_idx", "features_vec")
