import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any, Tuple

# PyTorch imports removed as NeuralRecommender is no longer used
# import torch 
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader, random_split

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
# from sklearn.linear_model import LogisticRegression # Removed
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier # Added for DecisionTreeRecommender
from pyspark.sql import DataFrame, Window, SparkSession
from pyspark.sql import functions as sf
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.types import DoubleType, ArrayType, FloatType, StructType, StructField

# Initialize Spark session
spark = SparkSession.builder.appName("ContentBasedRecommenders").getOrCreate()

class FeatureProcessor:
    """Helper class to process user and item features"""
    def __init__(self):
        self.user_scaler = StandardScaler()
        self.item_scaler = StandardScaler()
        self._user_scaler_fitted = False
        self._item_scaler_fitted = False
        
    def process_features(self, df: DataFrame, id_col: str, feature_cols: List[str], is_user: bool = True) -> Tuple[pd.Series, np.ndarray]:
        """Process features by converting to dense vectors and scaling.
        Returns a tuple of (pandas Series of IDs, numpy ndarray of scaled features).
        """
        # Select ID and feature columns
        pdf = df.select([id_col] + feature_cols).toPandas()
        ids_pd = pdf[id_col]
        features_pd = pdf[feature_cols]
        
        # Scale features
        scaler = self.user_scaler if is_user else self.item_scaler
        scaler_fitted = self._user_scaler_fitted if is_user else self._item_scaler_fitted

        if not scaler_fitted:
            scaled_features_np = scaler.fit_transform(features_pd)
            if is_user:
                self._user_scaler_fitted = True
            else:
                self._item_scaler_fitted = True
        else:
            scaled_features_np = scaler.transform(features_pd)
        return ids_pd, scaled_features_np

class PriceAwareRanking:
    """Mixin class for price-aware ranking"""
    def adjust_relevance_with_price(self, recs: DataFrame, price_col: str = "price", relevance_col: str = "relevance", price_weight: float = 0.3) -> DataFrame:
        """Adjust relevance scores based on item prices"""
        # Ensure recs has price and relevance columns
        if price_col not in recs.columns or relevance_col not in recs.columns:
            # If price is not available, return original relevance
            print(f"Warning: Price column '{price_col}' or relevance column '{relevance_col}' not found in recommendations. Skipping price adjustment.")
            return recs

        recs_with_price = recs.filter(sf.col(price_col).isNotNull())
        recs_without_price = recs.filter(sf.col(price_col).isNull())

        if recs_with_price.count() == 0:
             print(f"Warning: Price column '{price_col}' has no non-null values. Skipping price adjustment.")
             return recs


        price_stats = recs_with_price.agg(
            sf.min(price_col).alias("min_price"),
            sf.max(price_col).alias("max_price")
        ).collect()[0]
        
        min_price, max_price = price_stats["min_price"], price_stats["max_price"]
        
        # Avoid division by zero if all prices are the same
        if min_price is None or max_price is None : # Should be caught by filter isnull
             print(f"Warning: Min/Max price is None. Skipping price adjustment.")
             return recs # Should not happen if recs_with_price is not empty

        price_range = max_price - min_price
        
        if price_range > 1e-6: # Use a small epsilon for float comparison
            # Normalize price and create price score (higher price = higher score, assuming price is positive)
            # If items can have negative price (e.g. discounts making them free), this normalization might need adjustment
            recs_with_price = recs_with_price.withColumn(
                "price_score",
                (sf.col(price_col) - min_price) / price_range
            )
            
            # Combine relevance and price scores
            recs_with_price = recs_with_price.withColumn(
                relevance_col, # Overwrite existing relevance
                (1 - price_weight) * sf.col(relevance_col) + price_weight * sf.col("price_score")
            ).drop("price_score")
        # If price_range is zero (all items have same price), no adjustment needed based on price diversity
        
        return recs_with_price.unionByName(recs_without_price)


class KNNRecommender(PriceAwareRanking):
    """User-Based K-Nearest Neighbors based recommender system using content features."""
    def __init__(self, n_similar_users: int = 10, metric: str = 'cosine', seed: Optional[int] = None):
        self.n_similar_users = n_similar_users
        self.metric = metric
        self.seed = seed # seed is not used in current KNN logic directly, but good to keep
        self.feature_processor = FeatureProcessor()
        
        self.user_knn_model = None
        self.training_user_ids_pd: Optional[pd.Series] = None
        self.training_user_features_np: Optional[np.ndarray] = None
        self.training_log_pd: Optional[pd.DataFrame] = None
        
    def fit(self, log: DataFrame, user_features: Optional[DataFrame] = None, 
            item_features: Optional[DataFrame] = None): # item_features not used by this user-based KNN
        """Fit the KNN model using user features and interaction log."""
        if user_features is None:
            print("KNNRecommender: User features are required for fitting. Skipping fit.")
            return
        if log is None or log.count() == 0:
            print("KNNRecommender: Interaction log is required for fitting. Skipping fit.")
            return

        user_feature_cols = [col for col in user_features.columns if col.startswith('user_attr_')]
        if not user_feature_cols:
            print("KNNRecommender: No 'user_attr_' columns found in user_features. Skipping fit.")
            return
            
        self.training_user_ids_pd, self.training_user_features_np = \
            self.feature_processor.process_features(user_features, 'user_idx', user_feature_cols, is_user=True)
        
        if self.training_user_features_np.shape[0] < self.n_similar_users:
            print(f"KNNRecommender: Number of users for fitting ({self.training_user_features_np.shape[0]}) is less than n_similar_users ({self.n_similar_users}). Adjusting n_similar_users.")
            self.n_similar_users = max(1, self.training_user_features_np.shape[0]) # Ensure at least 1

        if self.training_user_features_np.shape[0] > 0 :
            self.user_knn_model = NearestNeighbors(n_neighbors=self.n_similar_users, metric=self.metric)
            self.user_knn_model.fit(self.training_user_features_np)
        else:
            print("KNNRecommender: No users to fit the KNN model. Skipping fit.")
            return # Cannot proceed without a fitted model

        # Store interaction log as pandas for faster lookup with numpy indices
        # Ensure log contains user_idx and item_idx
        if "user_idx" not in log.columns or "item_idx" not in log.columns:
            print("KNNRecommender: Log must contain 'user_idx' and 'item_idx'. Skipping fit.")
            self.user_knn_model = None # Invalidate model
            return
        self.training_log_pd = log.select("user_idx", "item_idx").toPandas()
    
    def predict(self, log: DataFrame, k: int, users: DataFrame, items: DataFrame, # Changed users_df to users, items_df to items
                user_features: Optional[DataFrame] = None, item_features: Optional[DataFrame] = None, 
                filter_seen_items: bool = True) -> DataFrame:
        """Generate recommendations using User-Based Content KNN."""

        if self.user_knn_model is None or self.training_user_ids_pd is None or self.training_log_pd is None:
            print("KNNRecommender: Model not fitted. Returning empty recommendations or random if specified elsewhere.")
            # Fallback: recommend random items, or handle as per competition rules for unfitted models
            return users.crossJoin(items.select("item_idx", "price")).withColumn("relevance", sf.rand(seed=self.seed)) # Use users, items


        if user_features is None:
            print("KNNRecommender: User features are required for prediction. Returning empty.")
            return spark.createDataFrame([], schema=users.schema.add("item_idx", items.schema["item_idx"].dataType).add("relevance", DoubleType())) # Use users, items


        user_feature_cols = [col for col in user_features.columns if col.startswith('user_attr_')]
        if not user_feature_cols:
            print("KNNRecommender: No 'user_attr_' columns found in user_features for prediction. Returning empty.")
            return spark.createDataFrame([], schema=users.schema.add("item_idx", items.schema["item_idx"].dataType).add("relevance", DoubleType()))

        # Process features for the users we need to predict for.
        # These features should be scaled using the same scaler fitted on training_user_features.
        pred_user_ids_all_pd, pred_user_features_all_np = \
            self.feature_processor.process_features(user_features, 'user_idx', user_feature_cols, is_user=True)

        # Create a mapping from all prediction user_idx to their np array index
        pred_user_id_to_idx_map = {user_id: i for i, user_id in enumerate(pred_user_ids_all_pd)}
        
        # Get the list of target user_idx values from the input users DataFrame
        target_user_ids_list = [row.user_idx for row in users.select("user_idx").collect()] # Use users

        recommendations = [] # List of (user_idx, item_idx, relevance)

        for target_user_id in target_user_ids_list:
            if target_user_id not in pred_user_id_to_idx_map:
                # This user_id from users DataFrame was not in the user_features provided to predict
                continue 
            
            target_user_np_idx = pred_user_id_to_idx_map[target_user_id]
            target_user_feature_vector = pred_user_features_all_np[target_user_np_idx].reshape(1, -1)
            
            # Find similar users from the training set
            # distances: float ndarray of shape (n_queries, n_neighbors)
            # indices: int ndarray of shape (n_queries, n_neighbors)
            distances, neighbor_indices_in_training = self.user_knn_model.kneighbors(target_user_feature_vector)
            
            # Aggregate item scores from similar users
            item_scores: Dict[Any, float] = {}
            if distances.shape[1] == 0 : # No neighbors found (e.g. if n_similar_users was too small or dataset issue)
                continue

            for i in range(neighbor_indices_in_training.shape[1]):
                neighbor_original_idx = neighbor_indices_in_training[0, i]
                neighbor_user_id = self.training_user_ids_pd.iloc[neighbor_original_idx]
                similarity_weight = 1.0 / (1.0 + distances[0, i]) # Example: inverse distance as weight
                
                # Get items interacted by this similar neighbor
                neighbor_items = self.training_log_pd[self.training_log_pd['user_idx'] == neighbor_user_id]['item_idx']
                for item_id in neighbor_items:
                    item_scores[item_id] = item_scores.get(item_id, 0.0) + similarity_weight
            
            for item_id, score in item_scores.items():
                recommendations.append((target_user_id, item_id, float(score)))

        if not recommendations:
            # Fallback if no recommendations could be generated (e.g. no similar users found items)
            return users.crossJoin(items.select("item_idx", "price")).withColumn("relevance", sf.rand(seed=self.seed)) # Use users, items


        recs_spark_df = spark.createDataFrame(recommendations, ["user_idx", "item_idx", "relevance"])

        # Add price for price-aware ranking. items DataFrame should have item_idx and price.
        recs_spark_df = recs_spark_df.join(items.select("item_idx", "price"), "item_idx", "left") # Use items
        
        recs_spark_df = self.adjust_relevance_with_price(recs_spark_df) # Default price_col="price", relevance_col="relevance"

        # Filter seen items from the *current* iteration's log
        if filter_seen_items and log is not None and log.count() > 0 :
            if "user_idx" in log.columns and "item_idx" in log.columns:
                seen_items = log.select("user_idx", "item_idx").distinct()
                recs_spark_df = recs_spark_df.join(
                    seen_items,
                    on=["user_idx", "item_idx"],
                    how="left_anti"
                )
            else:
                print("KNNRecommender: 'user_idx' or 'item_idx' not in log for filtering seen items. Skipping filtering.")
        
        # Rank and return top-k
        window = Window.partitionBy("user_idx").orderBy(sf.desc("relevance"))
        recs_spark_df = recs_spark_df.withColumn("rank", sf.row_number().over(window))
        recs_spark_df = recs_spark_df.filter(sf.col("rank") <= k).drop("rank")
        
        # Ensure all users from users DataFrame are present, even if with no recommendations
        # This part needs to be completed if strict output schema matching for all input users is needed.
        # For now, it returns recs for users it could process.
        # A left join from `users` to `recs_spark_df` and filling NaNs could achieve this.
        
        # Ensure `price` column exists, fill with a default if not.
        if "price" not in recs_spark_df.columns:
            recs_spark_df = recs_spark_df.withColumn("price", sf.lit(0.0)) # Add default price if missing

        return recs_spark_df.select("user_idx", "item_idx", "relevance", "price")

class RandomForestRecommender(PriceAwareRanking):
    """Random Forest based recommender system for predicting purchase likelihood."""
    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = None, 
                 min_samples_leaf: int = 1, max_features: Any = 'sqrt', 
                 seed: Optional[int] = None, price_col_name: str = 'price'):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.seed = seed
        self.price_col_name = price_col_name
        self.feature_processor = FeatureProcessor()
        self.model: Optional[RandomForestClassifier] = None

        if self.seed is not None:
            np.random.seed(self.seed)

    def _prepare_rf_data(self, log: DataFrame, 
                           user_features_df: DataFrame, item_features_df: DataFrame, 
                           is_train: bool) -> Optional[Tuple[pd.DataFrame, Optional[pd.Series], Optional[pd.DataFrame]]]:
        """Helper to prepare data for Random Forest training or prediction.
           This is very similar to _prepare_logreg_data and can be refactored later if needed.
        """
        user_feature_cols = [col for col in user_features_df.columns if col.startswith('user_attr_')]
        item_feature_cols = [col for col in item_features_df.columns if col.startswith('item_attr_')]

        if not user_feature_cols and not item_feature_cols:
            print("RandomForestRecommender: No 'user_attr_' or 'item_attr_' feature columns. Need at least one set.")
            return None

        user_feats_df = None
        if user_feature_cols:
            user_ids_pd, user_feats_np = self.feature_processor.process_features(
                user_features_df, 'user_idx', user_feature_cols, is_user=True)
            user_feats_df = pd.DataFrame(user_feats_np, columns=[f"u_{col}" for col in user_feature_cols], index=user_ids_pd)

        item_feats_df = None
        if item_feature_cols:
            item_ids_pd, item_feats_np = self.feature_processor.process_features(
                item_features_df, 'item_idx', item_feature_cols, is_user=False)
            item_feats_df = pd.DataFrame(item_feats_np, columns=[f"i_{col}" for col in item_feature_cols], index=item_ids_pd)
        
        feature_dfs_to_merge = []
        if user_feats_df is not None: feature_dfs_to_merge.append((user_feats_df, 'user_idx'))
        if item_feats_df is not None: feature_dfs_to_merge.append((item_feats_df, 'item_idx'))

        if not feature_dfs_to_merge: return None

        if is_train:
            # Logic for handling price and relevance for training data
            if self.price_col_name in item_features_df.columns:
                 log_with_price_and_relevance = log.join(
                    item_features_df.select('item_idx', self.price_col_name), 'item_idx', 'left_outer'
                )
            else:
                log_with_price_and_relevance = log
                print(f"Warning (RF Train): price_col '{self.price_col_name}' not in item_features.")

            cols_for_training = ['user_idx', 'item_idx']
            if 'relevance' in log_with_price_and_relevance.columns: cols_for_training.append('relevance')
            else: print("Warning (RF Train): 'relevance' not in log for label creation.")

            training_data_pd = log_with_price_and_relevance.select(*cols_for_training).toPandas()
            training_data_pd['label'] = training_data_pd['relevance'].apply(lambda x: 1 if x is not None and x > 0 else 0) if 'relevance' in training_data_pd else 1
            
            data = training_data_pd
            for feat_df, id_col_key in feature_dfs_to_merge:
                data = data.merge(feat_df, on=id_col_key, how='left')
            
            check_cols_for_na = []
            if user_feats_df is not None and not user_feats_df.empty: check_cols_for_na.append(user_feats_df.columns[0])
            if item_feats_df is not None and not item_feats_df.empty: check_cols_for_na.append(item_feats_df.columns[0])
            if check_cols_for_na: data.dropna(subset=check_cols_for_na, inplace=True)
            
            if data.empty: return None

            labels = data['label']
            columns_to_drop = ['user_idx', 'item_idx', 'label']
            if 'relevance' in data.columns: columns_to_drop.append('relevance')
            features = data.drop(columns=columns_to_drop)
            return features, labels, None
        else: # Prediction
            pred_data_pd = log.select('user_idx', 'item_idx').toPandas()
            data = pred_data_pd
            for feat_df, id_col_key in feature_dfs_to_merge:
                data = data.merge(feat_df, on=id_col_key, how='left')

            original_pairs = data[['user_idx', 'item_idx']].copy()
            check_cols_for_na_pred = []
            if user_feats_df is not None and not user_feats_df.empty: check_cols_for_na_pred.append(user_feats_df.columns[0])
            if item_feats_df is not None and not item_feats_df.empty: check_cols_for_na_pred.append(item_feats_df.columns[0])
            if check_cols_for_na_pred: data.dropna(subset=check_cols_for_na_pred, inplace=True)
            
            if data.empty:
                print("RandomForestRecommender: No data after merging features for prediction.")
                return None, None, original_pairs

            features = data.drop(columns=['user_idx', 'item_idx'])
            return features, None, original_pairs

    def fit(self, log: DataFrame, user_features: Optional[DataFrame] = None,
            item_features: Optional[DataFrame] = None):
        if log is None or log.count() == 0 or user_features is None or item_features is None:
            print("RandomForestRecommender: Missing data for fitting. Skipping.")
            return

        prepared_data = self._prepare_rf_data(log, user_features, item_features, is_train=True)
        if prepared_data is None:
            print("RandomForestRecommender: Data preparation for training failed.")
            return
        
        X, y, _ = prepared_data
        if X.empty or y is None or y.empty:
            print("RandomForestRecommender: No training data after preparation.")
            return
        
        if y.nunique() < 2:
            print(f"RandomForestRecommender: Only one class ({y.unique()}) present. Skipping fit.")
            return

        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.seed,
            class_weight='balanced' # Good for potentially imbalanced purchase data
        )
        try:
            self.model.fit(X, y)
            print(f"RandomForestRecommender: Trained model with n_estimators={self.n_estimators}.")
        except Exception as e:
            print(f"RandomForestRecommender: Error during model fitting: {e}")
            self.model = None

    def predict(self, log: DataFrame, k: int, users: DataFrame, items: DataFrame, 
                user_features: Optional[DataFrame] = None, item_features: Optional[DataFrame] = None,
                filter_seen_items: bool = True) -> DataFrame:
        if self.model is None:
            print("RandomForestRecommender: Model not fitted. Returning random.")
            items_with_price = items.selectExpr("item_idx", f"coalesce({self.price_col_name}, 0.0) as {self.price_col_name}")
            return users.crossJoin(items_with_price).withColumn("relevance", sf.rand(seed=self.seed))

        if user_features is None or item_features is None:
            print("RandomForestRecommender: Missing features for prediction. Returning random.")
            items_with_price = items.selectExpr("item_idx", f"coalesce({self.price_col_name}, 0.0) as {self.price_col_name}")
            return users.crossJoin(items_with_price).withColumn("relevance", sf.rand(seed=self.seed))

        items_for_pred = items
        if self.price_col_name not in items.columns:
            items_for_pred = items.withColumn(self.price_col_name, sf.lit(None).cast(DoubleType()))
        
        user_item_pairs_to_predict_df = users.crossJoin(items_for_pred.select("item_idx"))

        prepared_data = self._prepare_rf_data(user_item_pairs_to_predict_df, user_features, item_features, is_train=False)

        if prepared_data is None or prepared_data[0] is None:
            print("RandomForestRecommender: Data prep for prediction failed. Returning random.")
            items_with_price = items.selectExpr("item_idx", f"coalesce({self.price_col_name}, 0.0) as {self.price_col_name}")
            return users.crossJoin(items_with_price).withColumn("relevance", sf.rand(seed=self.seed))

        X_pred_features, _, original_pairs_pd = prepared_data
        
        if X_pred_features.empty or original_pairs_pd is None or original_pairs_pd.empty:
            print("RandomForestRecommender: No data to predict on. Returning random.")
            items_with_price = items.selectExpr("item_idx", f"coalesce({self.price_col_name}, 0.0) as {self.price_col_name}")
            return users.crossJoin(items_with_price).withColumn("relevance", sf.rand(seed=self.seed))

        predictions_proba = self.model.predict_proba(X_pred_features)[:, 1]
        results_pd = original_pairs_pd.copy()
        results_pd['relevance'] = predictions_proba
        
        if self.price_col_name in item_features.columns:
            item_prices_pd = item_features.select('item_idx', self.price_col_name).toPandas()
            results_pd = results_pd.merge(item_prices_pd, on='item_idx', how='left')
        else:
            results_pd[self.price_col_name] = 0.0

        user_idx_type = users.schema["user_idx"].dataType
        item_idx_type = items.schema["item_idx"].dataType
        price_spark_type = item_features.schema[self.price_col_name].dataType if self.price_col_name in item_features.columns else DoubleType()

        result_schema = StructType([
            StructField("user_idx", user_idx_type, True),
            StructField("item_idx", item_idx_type, True),
            StructField(self.price_col_name, price_spark_type, True),
            StructField("relevance", DoubleType(), True)
        ])
        results_pd[self.price_col_name] = results_pd[self.price_col_name].fillna(0.0)
        # Ensure the pandas DataFrame columns are in the same order as the schema for robustness
        recs_df = spark.createDataFrame(results_pd[['user_idx', 'item_idx', self.price_col_name, 'relevance']], schema=result_schema)
        
        recs_df = self.adjust_relevance_with_price(recs_df)

        if filter_seen_items and log is not None and log.count() > 0:
            if "user_idx" in log.columns and "item_idx" in log.columns:
                seen_items = log.select("user_idx", "item_idx").distinct()
                recs_df = recs_df.join(seen_items, on=["user_idx", "item_idx"], how="left_anti")
        
        window_spec = Window.partitionBy("user_idx").orderBy(sf.desc("relevance"))
        recs_df = recs_df.withColumn("rank", sf.row_number().over(window_spec))
        recs_df = recs_df.filter(sf.col("rank") <= k).drop("rank")
        
        # Explicitly select the columns to ensure the price column is present and named correctly
        return recs_df.select(
            sf.col("user_idx"),
            sf.col("item_idx"),
            sf.col("relevance"),
            sf.col(self.price_col_name).alias(self.price_col_name) # Ensure correct name
        )

class DecisionTreeRecommender(PriceAwareRanking):
    """Decision Tree based recommender system for predicting purchase likelihood."""
    def __init__(self, criterion: str = 'gini', splitter: str = 'best', 
                 max_depth: Optional[int] = None, min_samples_split: int = 2,
                 min_samples_leaf: int = 1, max_features: Any = None, 
                 seed: Optional[int] = None, price_col_name: str = 'price'):
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.seed = seed
        self.price_col_name = price_col_name
        self.feature_processor = FeatureProcessor()
        self.model: Optional[DecisionTreeClassifier] = None

        if self.seed is not None:
            np.random.seed(self.seed) # For other numpy ops if needed

    def _prepare_dt_data(self, log: DataFrame, 
                           user_features_df: DataFrame, item_features_df: DataFrame, 
                           is_train: bool) -> Optional[Tuple[pd.DataFrame, Optional[pd.Series], Optional[pd.DataFrame]]]:
        """Helper to prepare data for Decision Tree training or prediction.
           This is very similar to _prepare_rf_data.
        """
        user_feature_cols = [col for col in user_features_df.columns if col.startswith('user_attr_')]
        item_feature_cols = [col for col in item_features_df.columns if col.startswith('item_attr_')]

        if not user_feature_cols and not item_feature_cols:
            print("DecisionTreeRecommender: No 'user_attr_' or 'item_attr_' feature columns. Need at least one set.")
            return None

        user_feats_df = None
        if user_feature_cols:
            user_ids_pd, user_feats_np = self.feature_processor.process_features(
                user_features_df, 'user_idx', user_feature_cols, is_user=True)
            user_feats_df = pd.DataFrame(user_feats_np, columns=[f"u_{col}" for col in user_feature_cols], index=user_ids_pd)

        item_feats_df = None
        if item_feature_cols:
            item_ids_pd, item_feats_np = self.feature_processor.process_features(
                item_features_df, 'item_idx', item_feature_cols, is_user=False)
            item_feats_df = pd.DataFrame(item_feats_np, columns=[f"i_{col}" for col in item_feature_cols], index=item_ids_pd)
        
        feature_dfs_to_merge = []
        if user_feats_df is not None: feature_dfs_to_merge.append((user_feats_df, 'user_idx'))
        if item_feats_df is not None: feature_dfs_to_merge.append((item_feats_df, 'item_idx'))

        if not feature_dfs_to_merge: return None

        if is_train:
            if self.price_col_name in item_features_df.columns:
                 log_with_price_and_relevance = log.join(
                    item_features_df.select('item_idx', self.price_col_name), 'item_idx', 'left_outer'
                )
            else:
                log_with_price_and_relevance = log
                print(f"Warning (DT Train): price_col '{self.price_col_name}' not in item_features.")

            cols_for_training = ['user_idx', 'item_idx']
            if 'relevance' in log_with_price_and_relevance.columns: cols_for_training.append('relevance')
            else: print("Warning (DT Train): 'relevance' not in log for label creation.")

            training_data_pd = log_with_price_and_relevance.select(*cols_for_training).toPandas()
            training_data_pd['label'] = training_data_pd['relevance'].apply(lambda x: 1 if x is not None and x > 0 else 0) if 'relevance' in training_data_pd else 1
            
            data = training_data_pd
            for feat_df, id_col_key in feature_dfs_to_merge:
                data = data.merge(feat_df, on=id_col_key, how='left')
            
            check_cols_for_na = []
            if user_feats_df is not None and not user_feats_df.empty: check_cols_for_na.append(user_feats_df.columns[0])
            if item_feats_df is not None and not item_feats_df.empty: check_cols_for_na.append(item_feats_df.columns[0])
            if check_cols_for_na: data.dropna(subset=check_cols_for_na, inplace=True)
            
            if data.empty: 
                print("DecisionTreeRecommender: No data after merging features for training.")
                return None

            labels = data['label']
            columns_to_drop = ['user_idx', 'item_idx', 'label']
            if 'relevance' in data.columns: columns_to_drop.append('relevance')
            features = data.drop(columns=columns_to_drop)
            return features, labels, None
        else: # Prediction
            pred_data_pd = log.select('user_idx', 'item_idx').toPandas()
            data = pred_data_pd
            for feat_df, id_col_key in feature_dfs_to_merge:
                data = data.merge(feat_df, on=id_col_key, how='left')

            original_pairs = data[['user_idx', 'item_idx']].copy()
            check_cols_for_na_pred = []
            if user_feats_df is not None and not user_feats_df.empty: check_cols_for_na_pred.append(user_feats_df.columns[0])
            if item_feats_df is not None and not item_feats_df.empty: check_cols_for_na_pred.append(item_feats_df.columns[0])
            if check_cols_for_na_pred: data.dropna(subset=check_cols_for_na_pred, inplace=True)
            
            if data.empty:
                print("DecisionTreeRecommender: No data after merging features for prediction.")
                return None, None, original_pairs

            features = data.drop(columns=['user_idx', 'item_idx'])
            return features, None, original_pairs

    def fit(self, log: DataFrame, user_features: Optional[DataFrame] = None,
            item_features: Optional[DataFrame] = None):
        if log is None or log.count() == 0 or user_features is None or item_features is None:
            print("DecisionTreeRecommender: Missing data for fitting. Skipping.")
            return

        prepared_data = self._prepare_dt_data(log, user_features, item_features, is_train=True)
        if prepared_data is None:
            print("DecisionTreeRecommender: Data preparation for training failed.")
            return
        
        X, y, _ = prepared_data
        if X.empty or y is None or y.empty:
            print("DecisionTreeRecommender: No training data after preparation.")
            return
        
        if y.nunique() < 2:
            print(f"DecisionTreeRecommender: Only one class ({y.unique()}) present. Skipping fit.")
            return

        self.model = DecisionTreeClassifier(
            criterion=self.criterion,
            splitter=self.splitter,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.seed,
            class_weight='balanced' 
        )
        try:
            self.model.fit(X, y)
            print(f"DecisionTreeRecommender: Trained model with max_depth={self.max_depth}.")
        except Exception as e:
            print(f"DecisionTreeRecommender: Error during model fitting: {e}")
            self.model = None

    def predict(self, log: DataFrame, k: int, users: DataFrame, items: DataFrame, 
                user_features: Optional[DataFrame] = None, item_features: Optional[DataFrame] = None,
                filter_seen_items: bool = True) -> DataFrame:
        if self.model is None:
            print("DecisionTreeRecommender: Model not fitted. Returning random.")
            items_with_price = items.selectExpr("item_idx", f"coalesce({self.price_col_name}, 0.0) as {self.price_col_name}")
            return users.crossJoin(items_with_price).withColumn("relevance", sf.rand(seed=self.seed))

        if user_features is None or item_features is None:
            print("DecisionTreeRecommender: Missing features for prediction. Returning random.")
            items_with_price = items.selectExpr("item_idx", f"coalesce({self.price_col_name}, 0.0) as {self.price_col_name}")
            return users.crossJoin(items_with_price).withColumn("relevance", sf.rand(seed=self.seed))

        items_for_pred = items
        if self.price_col_name not in items.columns:
            items_for_pred = items.withColumn(self.price_col_name, sf.lit(None).cast(DoubleType()))
        
        user_item_pairs_to_predict_df = users.crossJoin(items_for_pred.select("item_idx"))

        prepared_data = self._prepare_dt_data(user_item_pairs_to_predict_df, user_features, item_features, is_train=False)

        if prepared_data is None or prepared_data[0] is None:
            print("DecisionTreeRecommender: Data prep for prediction failed. Returning random.")
            items_with_price = items.selectExpr("item_idx", f"coalesce({self.price_col_name}, 0.0) as {self.price_col_name}")
            return users.crossJoin(items_with_price).withColumn("relevance", sf.rand(seed=self.seed))

        X_pred_features, _, original_pairs_pd = prepared_data
        
        if X_pred_features.empty or original_pairs_pd is None or original_pairs_pd.empty:
            print("DecisionTreeRecommender: No data to predict on. Returning random.")
            items_with_price = items.selectExpr("item_idx", f"coalesce({self.price_col_name}, 0.0) as {self.price_col_name}")
            return users.crossJoin(items_with_price).withColumn("relevance", sf.rand(seed=self.seed))

        predictions_proba = self.model.predict_proba(X_pred_features)[:, 1]
        results_pd = original_pairs_pd.copy()
        results_pd['relevance'] = predictions_proba
        
        if self.price_col_name in item_features.columns:
            item_prices_pd = item_features.select('item_idx', self.price_col_name).toPandas()
            results_pd = results_pd.merge(item_prices_pd, on='item_idx', how='left')
        else:
            results_pd[self.price_col_name] = 0.0

        user_idx_type = users.schema["user_idx"].dataType
        item_idx_type = items.schema["item_idx"].dataType
        price_spark_type = item_features.schema[self.price_col_name].dataType if self.price_col_name in item_features.columns else DoubleType()

        result_schema = StructType([
            StructField("user_idx", user_idx_type, True),
            StructField("item_idx", item_idx_type, True),
            StructField(self.price_col_name, price_spark_type, True),
            StructField("relevance", DoubleType(), True)
        ])
        results_pd[self.price_col_name] = results_pd[self.price_col_name].fillna(0.0)
        # Ensure the pandas DataFrame columns are in the same order as the schema for robustness
        recs_df = spark.createDataFrame(results_pd[['user_idx', 'item_idx', self.price_col_name, 'relevance']], schema=result_schema)
        
        recs_df = self.adjust_relevance_with_price(recs_df)

        if filter_seen_items and log is not None and log.count() > 0:
            if "user_idx" in log.columns and "item_idx" in log.columns:
                seen_items = log.select("user_idx", "item_idx").distinct()
                recs_df = recs_df.join(seen_items, on=["user_idx", "item_idx"], how="left_anti")
        
        window_spec = Window.partitionBy("user_idx").orderBy(sf.desc("relevance"))
        recs_df = recs_df.withColumn("rank", sf.row_number().over(window_spec))
        recs_df = recs_df.filter(sf.col("rank") <= k).drop("rank")
        
        # Explicitly select the columns to ensure the price column is present and named correctly
        return recs_df.select(
            sf.col("user_idx"),
            sf.col("item_idx"),
            sf.col("relevance"),
            sf.col(self.price_col_name).alias(self.price_col_name) # Ensure correct name
        ) 