import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import shutil
from typing import List, Dict

# Cell: Import libraries and set up environment
"""
# Recommender Systems Analysis and Visualization
This notebook performs an exploratory analysis of recommender systems using the Sim4Rec library.
We'll generate synthetic data, compare multiple baseline recommenders, and visualize their performance.
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as sf
from pyspark.sql import DataFrame, Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.types import DoubleType, ArrayType

# Set up plotting
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)

# Initialize Spark session
spark = SparkSession.builder \
    .appName("RecSysVisualization") \
    .master("local[*]") \
    .config("spark.driver.memory", "6g") \
    .config("spark.sql.shuffle.partitions", "8") \
    .config("spark.speculation", "false") \
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

# Import our custom content-based recommenders
from content_based_recommenders import KNNRecommender, RandomForestRecommender, DecisionTreeRecommender
from my_recommender import MyRecommender

# Helper to average a metric from a list of metric dicts
def get_average_metric(metrics_history_list: List[Dict[str, float]], 
                       metric_name_in_dict: str, 
                       default_val: float = np.nan) -> float:
    if not metrics_history_list:
        return default_val
    
    values = []
    for iteration_metrics in metrics_history_list:
        if isinstance(iteration_metrics, dict):
            values.append(iteration_metrics.get(metric_name_in_dict, default_val))
        else:
            # Handle cases where an item in the list might not be a dict (should not happen with current simulator.py)
            values.append(default_val) 
            
    # Filter out default_val if it represents missing data before averaging
    valid_values = [v for v in values if v is not default_val and not (isinstance(v, float) and np.isnan(v))]
    
    if not valid_values: 
        return default_val
    return sum(valid_values) / len(valid_values)


# Import our custom content-based recommenders
from content_based_recommenders import KNNRecommender, RandomForestRecommender, DecisionTreeRecommender
from my_recommender import MyRecommender

# Helper to average a metric from a list of metric dicts
def get_average_metric(metrics_history_list: List[Dict[str, float]], 
                       metric_name_in_dict: str, 
                       default_val: float = np.nan) -> float:
    if not metrics_history_list:
        return default_val
    
    values = []
    for iteration_metrics in metrics_history_list:
        if isinstance(iteration_metrics, dict):
            values.append(iteration_metrics.get(metric_name_in_dict, default_val))
        else:
            # Handle cases where an item in the list might not be a dict (should not happen with current simulator.py)
            values.append(default_val) 
            
    # Filter out default_val if it represents missing data before averaging
    valid_values = [v for v in values if v is not default_val and not (isinstance(v, float) and np.isnan(v))]
    
    if not valid_values: 
        return default_val
    return sum(valid_values) / len(valid_values)


# Cell: Define custom recommender template
"""
## MyRecommender Template
Below is a template class for implementing a custom recommender system.
Students should extend this class with their own recommendation algorithm.
"""

# class MyRecommender:
#     """
#     Template class for implementing a custom recommender.
#     
#     This class provides the basic structure required to implement a recommender
#     that can be used with the Sim4Rec simulator. Students should extend this class
#     with their own recommendation algorithm.
#     """
#     
#     def __init__(self, seed=None):
#         """
#         Initialize recommender.
#         
#         Args:
#             seed: Random seed for reproducibility
#         """
#         self.seed = seed
#         # Add your initialization logic here
#     
#     def fit(self, log, user_features=None, item_features=None):
#         """
#         Train the recommender model based on interaction history.
#         
#         Args:
#             log: Interaction log with user_idx, item_idx, and relevance columns
#             user_features: User features dataframe (optional)
#             item_features: Item features dataframe (optional)
#         """
#         # Implement your training logic here
#         # For example:
#         #  1. Extract relevant features from user_features and item_features
#         #  2. Learn user preferences from the log
#         #  3. Build item similarity matrices or latent factor models
#         #  4. Store learned parameters for later prediction
#         pass
#     
#     def predict(self, log, k, users, items, user_features=None, item_features=None, filter_seen_items=True):
#         """
#         Generate recommendations for users.
#         
#         Args:
#             log: Interaction log with user_idx, item_idx, and relevance columns
#             k: Number of items to recommend
#             users: User dataframe
#             items: Item dataframe
#             user_features: User features dataframe (optional)
#             item_features: Item features dataframe (optional)
#             filter_seen_items: Whether to filter already seen items
#             
#         Returns:
#             DataFrame: Recommendations with user_idx, item_idx, and relevance columns
#         """
#         # Implement your recommendation logic here
#         # For example:
#         #  1. Extract relevant features for prediction
#         #  2. Calculate relevance scores for each user-item pair
#         #  3. Rank items by relevance and select top-k
#         #  4. Return a dataframe with columns: user_idx, item_idx, relevance
#         
#         # Example of a random recommender implementation:
#         # Cross join users and items
#         recs = users.crossJoin(items)
#         
#         # Filter out already seen items if needed
#         if filter_seen_items and log is not None:
#             seen_items = log.select("user_idx", "item_idx")
#             recs = recs.join(
#                 seen_items,
#                 on=["user_idx", "item_idx"],
#                 how="left_anti"
#             )
#         
#         # Add random relevance scores
#         recs = recs.withColumn(
#             "relevance",
#             sf.rand(seed=self.seed)
#         )
#         
#         # Rank items by relevance for each user
#         window = Window.partitionBy("user_idx").orderBy(sf.desc("relevance"))
#         recs = recs.withColumn("rank", sf.row_number().over(window))
#         
#         # Filter top-k recommendations
#         recs = recs.filter(sf.col("rank") <= k).drop("rank")
#         
#         return recs

# Cell: Data Exploration Functions
"""
## Data Exploration Functions
These functions help us understand the generated synthetic data.
"""

def explore_user_data(users_df):
    """
    Explore user data distributions and characteristics.
    
    Args:
        users_df: DataFrame containing user data
    """
    print("=== User Data Exploration ===")
    
    # Get basic statistics
    print(f"Total number of users: {users_df.count()}")
    
    # User segments distribution
    segment_counts = users_df.groupBy("segment").count().toPandas()
    print("\nUser Segments Distribution:")
    for _, row in segment_counts.iterrows():
        print(f"  {row['segment']}: {row['count']} users ({row['count']/users_df.count()*100:.1f}%)")
    
    # Plot user segments
    plt.figure(figsize=(10, 6))
    plt.pie(segment_counts['count'], labels=segment_counts['segment'], autopct='%1.1f%%', startangle=90, shadow=True)
    plt.title('User Segments Distribution')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('user_segments_distribution.png')
    print("User segments visualization saved to 'user_segments_distribution.png'")
    
    # Convert to pandas for easier feature analysis
    users_pd = users_df.toPandas()
    
    # Analyze user feature distributions
    feature_cols = [col for col in users_pd.columns if col.startswith('user_attr_')]
    if len(feature_cols) > 0:
        # Take a sample of feature columns if there are many
        sample_features = feature_cols[:min(5, len(feature_cols))]
        
        # Plot histograms for sample features
        plt.figure(figsize=(14, 8))
        for i, feature in enumerate(sample_features):
            plt.subplot(2, 3, i+1)
            for segment in users_pd['segment'].unique():
                segment_data = users_pd[users_pd['segment'] == segment]
                plt.hist(segment_data[feature], alpha=0.5, bins=20, label=segment)
            plt.title(f'Distribution of {feature}')
            plt.xlabel('Value')
            plt.ylabel('Count')
            if i == 0:
                plt.legend()
        plt.tight_layout()
        plt.savefig('user_feature_distributions.png')
        print("User feature distributions saved to 'user_feature_distributions.png'")
        
        # Feature correlation heatmap
        plt.figure(figsize=(12, 10))
        corr = users_pd[feature_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, cmap='coolwarm', vmax=.3, center=0,
                    square=True, linewidths=.5, annot=False, fmt='.2f')
        plt.title('User Feature Correlations')
        plt.tight_layout()
        plt.savefig('user_feature_correlations.png')
        print("User feature correlations saved to 'user_feature_correlations.png'")


def explore_item_data(items_df):
    """
    Explore item data distributions and characteristics.
    
    Args:
        items_df: DataFrame containing item data
    """
    print("\n=== Item Data Exploration ===")
    
    # Get basic statistics
    print(f"Total number of items: {items_df.count()}")
    
    # Item categories distribution
    category_counts = items_df.groupBy("category").count().toPandas()
    print("\nItem Categories Distribution:")
    for _, row in category_counts.iterrows():
        print(f"  {row['category']}: {row['count']} items ({row['count']/items_df.count()*100:.1f}%)")
    
    # Plot item categories
    plt.figure(figsize=(10, 6))
    plt.pie(category_counts['count'], labels=category_counts['category'], autopct='%1.1f%%', startangle=90, shadow=True)
    plt.title('Item Categories Distribution')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('item_categories_distribution.png')
    print("Item categories visualization saved to 'item_categories_distribution.png'")
    
    # Convert to pandas for easier feature analysis
    items_pd = items_df.toPandas()
    
    # Analyze price distribution
    if 'price' in items_pd.columns:
        plt.figure(figsize=(14, 6))
        
        # Overall price distribution
        plt.subplot(1, 2, 1)
        plt.hist(items_pd['price'], bins=30, alpha=0.7)
        plt.title('Overall Price Distribution')
        plt.xlabel('Price')
        plt.ylabel('Count')
        
        # Price by category
        plt.subplot(1, 2, 2)
        for category in items_pd['category'].unique():
            category_data = items_pd[items_pd['category'] == category]
            plt.hist(category_data['price'], alpha=0.5, bins=20, label=category)
        plt.title('Price Distribution by Category')
        plt.xlabel('Price')
        plt.ylabel('Count')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('item_price_distributions.png')
        print("Item price distributions saved to 'item_price_distributions.png'")
    
    # Analyze item feature distributions
    feature_cols = [col for col in items_pd.columns if col.startswith('item_attr_')]
    if len(feature_cols) > 0:
        # Take a sample of feature columns if there are many
        sample_features = feature_cols[:min(5, len(feature_cols))]
        
        # Plot histograms for sample features
        plt.figure(figsize=(14, 8))
        for i, feature in enumerate(sample_features):
            plt.subplot(2, 3, i+1)
            for category in items_pd['category'].unique():
                category_data = items_pd[items_pd['category'] == category]
                plt.hist(category_data[feature], alpha=0.5, bins=20, label=category)
            plt.title(f'Distribution of {feature}')
            plt.xlabel('Value')
            plt.ylabel('Count')
            if i == 0:
                plt.legend()
        plt.tight_layout()
        plt.savefig('item_feature_distributions.png')
        print("Item feature distributions saved to 'item_feature_distributions.png'")


def explore_interactions(history_df, users_df, items_df):
    """
    Explore interaction patterns between users and items.
    
    Args:
        history_df: DataFrame containing interaction history
        users_df: DataFrame containing user data
        items_df: DataFrame containing item data
    """
    print("\n=== Interaction Data Exploration ===")
    
    # Get basic statistics
    total_interactions = history_df.count()
    total_users = users_df.count()
    total_items = items_df.count()
    
    print(f"Total interactions: {total_interactions}")
    print(f"Interaction density: {total_interactions / (total_users * total_items) * 100:.4f}%")
    
    # Users with interactions
    users_with_interactions = history_df.select("user_idx").distinct().count()
    print(f"Users with at least one interaction: {users_with_interactions} ({users_with_interactions/total_users*100:.1f}%)")
    
    # Items with interactions
    items_with_interactions = history_df.select("item_idx").distinct().count()
    print(f"Items with at least one interaction: {items_with_interactions} ({items_with_interactions/total_items*100:.1f}%)")
    
    # Distribution of interactions per user
    interactions_per_user = history_df.groupBy("user_idx").count().toPandas()
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(interactions_per_user['count'], bins=20)
    plt.title('Distribution of Interactions per User')
    plt.xlabel('Number of Interactions')
    plt.ylabel('Number of Users')
    
    # Distribution of interactions per item
    interactions_per_item = history_df.groupBy("item_idx").count().toPandas()
    
    plt.subplot(1, 2, 2)
    plt.hist(interactions_per_item['count'], bins=20)
    plt.title('Distribution of Interactions per Item')
    plt.xlabel('Number of Interactions')
    plt.ylabel('Number of Items')
    
    plt.tight_layout()
    plt.savefig('interaction_distributions.png')
    print("Interaction distributions saved to 'interaction_distributions.png'")
    
    # Analyze relevance distribution
    if 'relevance' in history_df.columns:
        relevance_dist = history_df.groupBy("relevance").count().toPandas()
        
        plt.figure(figsize=(10, 6))
        plt.bar(relevance_dist['relevance'].astype(str), relevance_dist['count'])
        plt.title('Distribution of Relevance Scores')
        plt.xlabel('Relevance Score')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig('relevance_distribution.png')
        print("Relevance distribution saved to 'relevance_distribution.png'")
    
    # If we have user segments and item categories, analyze cross-interactions
    if 'segment' in users_df.columns and 'category' in items_df.columns:
        # Join with user segments and item categories
        interaction_analysis = history_df.join(
            users_df.select('user_idx', 'segment'),
            on='user_idx'
        ).join(
            items_df.select('item_idx', 'category'),
            on='item_idx'
        )
        
        # Count interactions by segment and category
        segment_category_counts = interaction_analysis.groupBy('segment', 'category').count().toPandas()
        
        # Create a pivot table
        pivot_table = segment_category_counts.pivot(index='segment', columns='category', values='count').fillna(0)
        
        # Plot heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_table, annot=True, fmt='g', cmap='viridis')
        plt.title('Interactions Between User Segments and Item Categories')
        plt.tight_layout()
        plt.savefig('segment_category_interactions.png')
        print("Segment-category interactions saved to 'segment_category_interactions.png'")


# Cell: Recommender Analysis Function
"""
## Recommender System Analysis
This is the main function to run analysis of different recommender systems and visualize the results.
"""

def run_recommender_analysis():
    """
    Run the full recommender system analysis pipeline.
    
    This function orchestrates data generation, EDA, recommender evaluation,
    and visualization of results.
    """
    config = DEFAULT_CONFIG
    
    # Data Generation
    data_generator_config = config["data_generation"]
    main_data_generator = CompetitionDataGenerator(
        spark_session=spark, 
        **data_generator_config
    )
    users_df = main_data_generator.generate_users()
    items_df = main_data_generator.generate_items()
    initial_history_df = main_data_generator.generate_initial_history(
        interaction_density=data_generator_config["initial_history_density"]
    )
    
    # Setup Sim4Rec specific generators from the main data generator instance
    # These are used by the CompetitionSimulator if it relies on Sim4Rec's internal generation mechanisms
    # for simulating responses, beyond the initial log.
    user_sim4rec_generator, item_sim4rec_generator = main_data_generator.setup_data_generators()

    # Exploratory Data Analysis (EDA)
    print("\n=== Starting Exploratory Data Analysis ===")
    explore_user_data(users_df)
    explore_item_data(items_df)
    explore_interactions(initial_history_df, users_df, items_df)
    
    # Set simulation parameters
    train_iterations = config["simulation"]["train_iterations"] # Original
    test_iterations = config["simulation"]["test_iterations"]   # Original
    # train_iterations = 1  # Reduced for faster testing
    # test_iterations = 1   # Reduced for faster testing
    # print(f"INFO: Reduced iterations for faster testing: train={train_iterations}, test={test_iterations}")
    
    # <<<< TEMP MODIFICATION: Force retrain to False for faster testing >>>>
    config["simulation"]["retrain"] = False
    print(f"INFO: Overriding simulation config: retrain = {config['simulation']['retrain']}")
    # <<<< END TEMP MODIFICATION >>>>
    
    # Initialize lists to store results for plotting
    recommender_names = []
    avg_metrics_history = defaultdict(list)
    avg_revenues_history = []

    # Define recommenders to evaluate
    # Each entry is a tuple: (recommender_instance, name_string)
    # Ensure price_col_name matches what's used in the simulator and data generator if your models need it explicitly.
    recommenders_to_evaluate = [
        # Baseline Recommenders
        (RandomRecommender(seed=config["data_generation"]["seed"]), "Random"),
        (PopularityRecommender(seed=config["data_generation"]["seed"]), "Popularity"),
        (ContentBasedRecommender(seed=config["data_generation"]["seed"], price_col_name='price'), "ContentBased"), # Original sample

        # KNNRecommender Variation
        (KNNRecommender(n_similar_users=10, metric='cosine', seed=config["data_generation"]["seed"]), "KNN (k=10, cosine)"),

        # RandomForestRecommender Variation
        (RandomForestRecommender(seed=config["data_generation"]["seed"], n_estimators=50, max_depth=8, price_col_name='price'), "RF (N=50, D=8)"),

        # DecisionTreeRecommender Variation
        (DecisionTreeRecommender(seed=config["data_generation"]["seed"], max_depth=10, price_col_name='price'), "DT (D=10)"),

        # MyRecommender (Ensemble of KNN, RF, DT)
        (MyRecommender(seed=config["data_generation"]["seed"], price_col_name='price'), "MyEnsemble (KNN+RF+DT)")
    ]
    
    # Initialize recommenders with initial history
    # This step might be redundant if simulator's fit is called at the start of each train_test_split
    # However, it can be useful if a recommender needs a global fit before iterations.
    print("\nPerforming initial fit for all recommenders...")
    for recommender_instance, recommender_name in recommenders_to_evaluate:
        print(f"Initial fit for {recommender_name}...")
        try:
            recommender_instance.fit(log=initial_history_df, 
                                     user_features=users_df, 
                                     item_features=items_df)
        except Exception as e:
            print(f"Error during initial fit for {recommender_name}: {e}")
            import traceback
            traceback.print_exc()


    results_list = []
    
    for recommender_instance, recommender_name in recommenders_to_evaluate:
        print(f"\nEvaluating {recommender_name}:")
        
        # The CompetitionSimulator might not need a new DataGenerator instance for each recommender
        # if it primarily uses the initial log_df and the user/item generators for its simulation loop.
        # The key is that log_df passed to CompetitionSimulator is the *initial* state.

        current_simulator_data_dir = f"simulator_run_data_{recommender_name}"
        if os.path.exists(current_simulator_data_dir):
            shutil.rmtree(current_simulator_data_dir)
            print(f"Cleaned up old simulator data directory: {current_simulator_data_dir}")

        simulator = CompetitionSimulator(
            spark_session=spark,
            user_generator=user_sim4rec_generator, 
            item_generator=item_sim4rec_generator,
            log_df=initial_history_df.alias(f"initial_log_for_{recommender_name}"), # Use a fresh alias of the initial log
            data_dir=current_simulator_data_dir, # Pass a unique data directory
            conversion_noise_mean=config["simulation"]["conversion_noise_mean"],
            conversion_noise_std=config["simulation"]["conversion_noise_std"],
            seed=config["data_generation"]["seed"]
            # config object itself is not an argument here
        )
        
        # Run simulation with train-test split
        # Ensure recommender_name is an accepted param by train_test_split in simulator.py if used for internal dir naming
        train_metrics, test_metrics, train_revenue, test_revenue = simulator.train_test_split(
            recommender=recommender_instance,
            train_iterations=train_iterations,
            test_iterations=test_iterations,
            # Pass other necessary parameters from config["simulation"] if they are arguments to train_test_split
            user_frac=config["simulation"]["user_fraction"],
            k=config["simulation"]["k"],
            filter_seen_items=config["simulation"]["filter_seen_items"],
            retrain=config["simulation"]["retrain"]
            # recommender_name=recommender_name # This was causing issues if not in method signature
        )
        
        # Store results
        result_entry = {
            "name": recommender_name,
            "train_revenue_trajectory": train_revenue, # train_revenue is train_revenue_history
            "test_revenue_trajectory": test_revenue,   # test_revenue is test_revenue_history
            "train_total_revenue": sum(train_revenue) if train_revenue else 0,
            "test_total_revenue": sum(test_revenue) if test_revenue else 0,
            "train_avg_revenue": sum(train_revenue) / train_iterations if train_iterations > 0 and train_revenue else 0,
            "test_avg_revenue": sum(test_revenue) / test_iterations if test_iterations > 0 and test_revenue else 0,
            "train_metrics_history": train_metrics, # Store the list of dicts
            "test_metrics_history": test_metrics   # Store the list of dicts
        }

        # Calculate and store average for each defined metric
        for metric_key in EVALUATION_METRICS.keys(): 
            # The keys in EVALUATION_METRICS (e.g., 'precision_at_k') 
            # should match the keys in the dicts returned by RankingMetrics.evaluate()
            result_entry[f"train_avg_{metric_key}"] = get_average_metric(train_metrics, metric_key)
            result_entry[f"test_avg_{metric_key}"] = get_average_metric(test_metrics, metric_key)
        
        results_list.append(result_entry)

    results_df = pd.DataFrame(results_list)
    if not results_df.empty:
        results_df = results_df.sort_values(by="test_total_revenue", ascending=False)
    
    print("\nRecommender Evaluation Results (sorted by test revenue):")
    print(results_df.to_string())
    
    # Generate comparison plots
    recommender_names_for_plot = [name for _, name in recommenders_to_evaluate] # Get names from the tuples
    if not results_df.empty:
        visualize_recommender_performance(results_df, recommender_names_for_plot)
        visualize_detailed_metrics(results_df, recommender_names_for_plot)
    else:
        print("\nNo results to visualize.")
        
    return results_df


# Cell: Performance Visualization Functions
"""
## Performance Visualization Functions
These functions create visualizations for comparing recommender performance.
"""

def visualize_recommender_performance(results_df, recommender_names):
    """
    Visualize the performance of recommenders in terms of revenue and key metrics.
    
    Args:
        results_df: DataFrame with evaluation results
        recommender_names: List of recommender names
    """
    plt.figure(figsize=(16, 16))
    
    # Plot total revenue comparison
    plt.subplot(3, 2, 1)
    x = np.arange(len(recommender_names))
    width = 0.35
    plt.bar(x - width/2, results_df['train_total_revenue'], width, label='Training')
    plt.bar(x + width/2, results_df['test_total_revenue'], width, label='Testing')
    plt.xlabel('Recommender')
    plt.ylabel('Total Revenue')
    plt.title('Total Revenue Comparison')
    plt.xticks(x, results_df['name'])
    plt.legend()
    
    # Plot average revenue per iteration
    plt.subplot(3, 2, 2)
    plt.bar(x - width/2, results_df['train_avg_revenue'], width, label='Training')
    plt.bar(x + width/2, results_df['test_avg_revenue'], width, label='Testing')
    plt.xlabel('Recommender')
    plt.ylabel('Avg Revenue per Iteration')
    plt.title('Average Revenue Comparison')
    plt.xticks(x, results_df['name'])
    plt.legend()
    
    # Plot discounted revenue comparison (if available)
    plt.subplot(3, 2, 3)
    if 'train_avg_discounted_revenue' in results_df.columns and 'test_avg_discounted_revenue' in results_df.columns:
        plt.bar(x - width/2, results_df['train_avg_discounted_revenue'], width, label='Training')
        plt.bar(x + width/2, results_df['test_avg_discounted_revenue'], width, label='Testing')
        plt.xlabel('Recommender')
        plt.ylabel('Avg Discounted Revenue')
        plt.title('Discounted Revenue Comparison')
        plt.xticks(x, results_df['name'])
        plt.legend()
    
    # Plot revenue trajectories
    plt.subplot(3, 2, 4)
    markers = ['o', 's', 'D', '^']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, name in enumerate(results_df['name']):
        # Combined train and test trajectories
        train_revenue = results_df.iloc[i]['train_revenue_trajectory']
        test_revenue = results_df.iloc[i]['test_revenue_trajectory']
        
        # Check if revenue is a scalar (numpy.float64) or a list/array
        if isinstance(train_revenue, (float, np.float64, np.float32, int, np.integer)):
            train_revenue = [train_revenue]
        if isinstance(test_revenue, (float, np.float64, np.float32, int, np.integer)):
            test_revenue = [test_revenue]
            
        iterations = list(range(len(train_revenue))) + list(range(len(test_revenue)))
        revenues = train_revenue + test_revenue
        
        plt.plot(iterations, revenues, marker=markers[i % len(markers)], 
                 color=colors[i % len(colors)], label=name)
        
        # Add a vertical line to separate train and test
        if i == 0:  # Only add the line once
            plt.axvline(x=len(train_revenue)-0.5, color='k', linestyle='--', alpha=0.3, label='Train/Test Split')
    
    plt.xlabel('Iteration')
    plt.ylabel('Revenue')
    plt.title('Revenue Trajectory (Training → Testing)')
    plt.legend()
    
    # Plot ranking metrics comparison - Training
    plt.subplot(3, 2, 5)
    
    # Select metrics to include
    ranking_metrics_keys = ['precision_at_k', 'ndcg_at_k', 'mrr', 'hit_rate'] # These are keys from EVALUATION_METRICS
    # Filter for metrics that are actually present in results_df (with _avg_ prefix)
    plottable_ranking_metrics = [m for m in ranking_metrics_keys if f'train_avg_{m}' in results_df.columns]
    
    # Create bar groups
    bar_positions = np.arange(len(plottable_ranking_metrics))
    bar_width = 0.8 / len(results_df) if len(results_df) > 0 else 0.8
    
    for i, (_, row) in enumerate(results_df.iterrows()):
        model_name = row['name']
        offsets = (i - len(results_df)/2 + 0.5) * bar_width
        # metric_values = [row[f'train_{m}'] for m in plottable_ranking_metrics]
        metric_values = [row[f'train_avg_{m}'] for m in plottable_ranking_metrics]
        plt.bar(bar_positions + offsets, metric_values, bar_width, label=model_name, 
                color=colors[i % len(colors)], alpha=0.7)
    
    plt.xlabel('Metric')
    plt.ylabel('Value')
    plt.title('Ranking Metrics Comparison (Training Phase)')
    plt.xticks(bar_positions, [m.replace('_', ' ').title() for m in plottable_ranking_metrics])
    plt.legend()
    
    # Plot ranking metrics comparison - Testing
    plt.subplot(3, 2, 6)
    
    # Filter for metrics that are actually present in results_df (with _avg_ prefix)
    plottable_ranking_metrics_test = [m for m in ranking_metrics_keys if f'test_avg_{m}' in results_df.columns]

    # Get best-performing model
    best_model_idx = results_df['test_total_revenue'].idxmax() if not results_df.empty and 'test_total_revenue' in results_df.columns else None
    best_model_name = results_df.iloc[best_model_idx]['name'] if best_model_idx is not None else ""
    
    # Create bar groups
    bar_positions_test = np.arange(len(plottable_ranking_metrics_test))
    # bar_width is same as above
    
    for i, (_, row) in enumerate(results_df.iterrows()):
        model_name = row['name']
        offsets = (i - len(results_df)/2 + 0.5) * bar_width
        # metric_values = [row[f'test_{m}'] for m in plottable_ranking_metrics_test]
        metric_values = [row[f'test_avg_{m}'] for m in plottable_ranking_metrics_test]
        plt.bar(bar_positions_test + offsets, metric_values, bar_width, label=model_name, 
                color=colors[i % len(colors)],
                alpha=0.7 if model_name != best_model_name else 1.0)
    
    plt.xlabel('Metric')
    plt.ylabel('Value')
    plt.title('Ranking Metrics Comparison (Test Phase)')
    plt.xticks(bar_positions_test, [m.replace('_', ' ').title() for m in plottable_ranking_metrics_test])
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('recommender_performance_comparison.png')
    print("\nPerformance visualizations saved to 'recommender_performance_comparison.png'")


def visualize_detailed_metrics(results_df, recommender_names):
    """
    Create detailed visualizations for each metric and recommender.
    
    Args:
        results_df: DataFrame with evaluation results
        recommender_names: List of recommender names
    """
    # Create a figure for metric trajectories
    plt.figure(figsize=(16, 16))
    
    # Get all available metrics from the first recommender's first iteration
    all_metric_keys_from_history = []
    if not results_df.empty and 'train_metrics_history' in results_df.columns:
        first_train_metrics_history = results_df.iloc[0]['train_metrics_history']
        if first_train_metrics_history and isinstance(first_train_metrics_history, list) and len(first_train_metrics_history) > 0 and isinstance(first_train_metrics_history[0], dict):
            all_metric_keys_from_history = list(first_train_metrics_history[0].keys())
    
    # Select key metrics to visualize (these should match keys in the history dicts)
    # These are typically also the keys in EVALUATION_METRICS
    key_metrics_to_plot = ['revenue', 'discounted_revenue', 'precision_at_k', 'ndcg_at_k', 'mrr', 'hit_rate']
    # Filter for those that are actually available from the history
    key_metrics_to_plot = [m for m in key_metrics_to_plot if m in all_metric_keys_from_history]
    
    # Plot metric trajectories for each key metric
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', 'D', '^']
    
    for i, metric_plot_key in enumerate(key_metrics_to_plot):
        if i < 6:  # Limit to 6 metrics to avoid overcrowding
            plt.subplot(3, 2, i+1)
            
            for j, name_of_recommender in enumerate(results_df['name']):
                row = results_df[results_df['name'] == name_of_recommender].iloc[0]
                
                # Get metric values for training phase from the history list
                train_values = []
                if 'train_metrics_history' in row and isinstance(row['train_metrics_history'], list):
                    for train_iteration_metric_dict in row['train_metrics_history']:
                        if isinstance(train_iteration_metric_dict, dict) and metric_plot_key in train_iteration_metric_dict:
                            train_values.append(train_iteration_metric_dict[metric_plot_key])
                
                # Get metric values for testing phase from the history list
                test_values = []
                if 'test_metrics_history' in row and isinstance(row['test_metrics_history'], list):
                    for test_iteration_metric_dict in row['test_metrics_history']:
                        if isinstance(test_iteration_metric_dict, dict) and metric_plot_key in test_iteration_metric_dict:
                            test_values.append(test_iteration_metric_dict[metric_plot_key])
                
                # Plot training phase
                if train_values:
                    plt.plot(range(len(train_values)), train_values, 
                             marker=markers[j % len(markers)], 
                             color=colors[j % len(colors)],
                             linestyle='-', label=f"{name_of_recommender} (Train)")
                
                # Plot testing phase (offsetting x-axis)
                if test_values:
                    test_x_offset = len(train_values) if train_values else 0
                    plt.plot(range(test_x_offset, test_x_offset + len(test_values)), test_values, 
                             marker=markers[j % len(markers)], 
                             color=colors[j % len(colors)],
                             linestyle='--', label=f"{name_of_recommender} (Test)")
            
            plt.title(f'{metric_plot_key.replace("_", " ").title()} Trajectory')
            plt.xlabel('Iteration (Train -> Test)')
            plt.ylabel(metric_plot_key.replace("_", " ").title())
            plt.legend()
    
    plt.tight_layout()
    plt.savefig('detailed_metric_trajectories.png')
    print("Detailed metrics visualizations saved to 'detailed_metric_trajectories.png'")
    
    # Create a correlation heatmap of metrics
    plt.figure(figsize=(14, 12))
    
    # Extract metrics columns for correlation
    # We want the single numeric value columns (total revenue, average revenue, average metrics)
    potential_metric_cols = [col for col in results_df.columns if col.startswith('train_') or col.startswith('test_')]
    metric_cols_for_corr = []
    for col in potential_metric_cols:
        if col.endswith('_trajectory') or col.endswith('_metrics_history'):
            continue # Skip list-based columns
        # Keep total revenue, average revenue, and all train_avg_METRIC / test_avg_METRIC
        if 'total_revenue' in col or 'avg_revenue' in col or col.startswith('train_avg_') or col.startswith('test_avg_'):
            metric_cols_for_corr.append(col)
            
    metric_cols_for_corr = sorted(list(set(metric_cols_for_corr))) # Ensure uniqueness and consistent order
    
    if len(metric_cols_for_corr) > 1:
        # Ensure all selected columns are indeed numeric before calling .corr()
        numeric_df_for_corr = results_df[metric_cols_for_corr].apply(pd.to_numeric, errors='coerce')
        # Drop any columns that could not be fully converted to numeric (e.g., if they unexpectedly contained non-numeric data)
        numeric_df_for_corr.dropna(axis=1, how='all', inplace=True)

        if len(numeric_df_for_corr.columns) > 1:
            correlation_df = numeric_df_for_corr.corr()
            
            # Plot heatmap
            sns.heatmap(correlation_df, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
            plt.title('Correlation Between Aggregated Metrics') # Updated title
            plt.tight_layout()
            plt.savefig('metrics_correlation_heatmap.png')
            print("Metrics correlation heatmap saved to 'metrics_correlation_heatmap.png'")
        else:
            print("Not enough numeric metric columns for correlation heatmap after coercion.")
    else:
        print("Not enough metric columns selected for correlation heatmap.")


def calculate_discounted_cumulative_gain(recommendations, k=5, discount_factor=0.85):
    """
    Calculate the Discounted Cumulative Gain for recommendations.
    
    Args:
        recommendations: DataFrame with recommendations (must have relevance column)
        k: Number of items to consider
        discount_factor: Factor to discount gains by position
        
    Returns:
        float: Average DCG across all users
    """
    # Group by user and calculate per-user DCG
    user_dcg = []
    for user_id, user_recs in recommendations.groupBy("user_idx").agg(
        sf.collect_list(sf.struct("relevance", "rank")).alias("recommendations")
    ).collect():
        # Sort by rank
        user_rec_list = sorted(user_id.recommendations, key=lambda x: x[1])
        
        # Calculate DCG
        dcg = 0
        for i, (rel, _) in enumerate(user_rec_list[:k]):
            # Apply discount based on position
            dcg += rel * (discount_factor ** i)
        
        user_dcg.append(dcg)
    
    # Return average DCG across all users
    return np.mean(user_dcg) if user_dcg else 0.0


# Cell: Main execution
"""
## Run the Analysis
When you run this notebook, it will perform the full analysis and visualization.
"""

if __name__ == "__main__":
    print("Script started...")
    results = run_recommender_analysis() 