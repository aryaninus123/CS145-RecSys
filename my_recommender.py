from typing import Optional, Dict, Any
import numpy as np
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as sf

from content_based_recommenders import (
    KNNRecommender,
    RandomForestRecommender,
    DecisionTreeRecommender
)

class MyRecommender:
    """
    Ensemble of content-based recommenders: KNN, RandomForest, and DecisionTree.
    """
    def __init__(self, seed: Optional[int] = None, price_col_name: str = 'price'):
        self.seed = seed
        np.random.seed(seed)
        self.price_col_name = price_col_name

        # Initialize recommenders
        self.knn = KNNRecommender(
            n_similar_users=5,
            metric='cosine',
            seed=seed
        )
        self.rf = RandomForestRecommender(
            n_estimators=30,
            max_depth=5,
            seed=seed,
            price_col_name=self.price_col_name
        )
        self.dt = DecisionTreeRecommender(
            max_depth=5,
            seed=seed,
            price_col_name=self.price_col_name
        )
        
        # Static model weights
        self.model_weights = {
            'knn': 0.34,
            'rf': 0.33,
            'dt': 0.33
        }
        
    def fit(self, log: DataFrame, user_features: Optional[DataFrame] = None,
            item_features: Optional[DataFrame] = None):
        """
        Fit all recommenders on the training data.
        """
        print("Fitting MyRecommender's KNN...")
        self.knn.fit(log, user_features, item_features)
        
        print("Fitting MyRecommender's RandomForest...")
        self.rf.fit(log, user_features, item_features)

        print("Fitting MyRecommender's DecisionTree...")
        self.dt.fit(log, user_features, item_features)
        
        print("MyRecommender fitting complete.")
    
    def predict(self, log: DataFrame, k: int, users: DataFrame, items: DataFrame,
                user_features: Optional[DataFrame] = None, item_features: Optional[DataFrame] = None,
                filter_seen_items: bool = True) -> DataFrame:
        """
        Generate recommendations by combining predictions from all models
        using static weights.
        """
        print("MyRecommender predicting with KNN...")
        knn_preds = self.knn.predict(
            log, k, users, items, user_features, item_features, filter_seen_items
        )
        
        print("MyRecommender predicting with RandomForest...")
        rf_preds = self.rf.predict(
            log, k, users, items, user_features, item_features, filter_seen_items
        )

        print("MyRecommender predicting with DecisionTree...")
        dt_preds = self.dt.predict(
            log, k, users, items, user_features, item_features, filter_seen_items
        )
        
        # Standardize relevance and price column names
        knn_preds = knn_preds.select(sf.col('user_idx'), sf.col('item_idx'), 
                                   sf.col('relevance').alias('knn_relevance'), 
                                   sf.col(self.price_col_name).alias('price_knn'))
        rf_preds = rf_preds.select(sf.col('user_idx'), sf.col('item_idx'), 
                                 sf.col('relevance').alias('rf_relevance'), 
                                 sf.col(self.price_col_name).alias('price_rf'))
        dt_preds = dt_preds.select(sf.col('user_idx'), sf.col('item_idx'), 
                                 sf.col('relevance').alias('dt_relevance'), 
                                 sf.col(self.price_col_name).alias('price_dt'))

        # Combine predictions: KNN outer join RF, then result outer join DT
        combined_kr = knn_preds.join(
            rf_preds,
            on=['user_idx', 'item_idx'],
            how='outer'
        )
        combined = combined_kr.join(
            dt_preds,
            on=['user_idx', 'item_idx'],
            how='outer'
        )
        
        # Coalesce price columns
        combined = combined.withColumn(
            self.price_col_name, 
            sf.coalesce(sf.col('price_knn'), sf.col('price_rf'), sf.col('price_dt'))
        ).drop('price_knn', 'price_rf', 'price_dt')

        # Fill NA relevance scores with 0 before weighting
        combined = combined.fillna(0, subset=['knn_relevance', 'rf_relevance', 'dt_relevance'])
        
        # Compute weighted average relevance
        combined = combined.withColumn(
            'relevance',
            (self.model_weights['knn'] * sf.col('knn_relevance')) +
            (self.model_weights['rf'] * sf.col('rf_relevance')) +
            (self.model_weights['dt'] * sf.col('dt_relevance'))
        ).select('user_idx', 'item_idx', 'relevance', self.price_col_name)
        
        # Rank and return top-k
        window = Window.partitionBy("user_idx").orderBy(sf.desc("relevance"))
        combined = combined.withColumn("rank", sf.row_number().over(window))
        combined = combined.filter(sf.col("rank") <= k).select('user_idx', 'item_idx', 'relevance', self.price_col_name)
        
        return combined
    