"""Synthetic data generation for model stealing detection."""

import hashlib
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import entropy


class ModelStealingDataGenerator:
    """
    Generate synthetic data for model stealing detection research.
    
    This class creates realistic query patterns that simulate both legitimate
    users and potential model stealers.
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize the data generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
    def generate_legitimate_queries(
        self,
        n_queries: int = 1000,
        n_features: int = 10,
        user_id: str = "legitimate_user"
    ) -> pd.DataFrame:
        """
        Generate legitimate user queries with normal distribution patterns.
        
        Args:
            n_queries: Number of queries to generate
            n_features: Number of features in each query
            user_id: User identifier
            
        Returns:
            pd.DataFrame: Generated queries with metadata
        """
        # Generate queries with normal distribution (typical user behavior)
        queries = np.random.normal(loc=0, scale=1, size=(n_queries, n_features))
        
        # Add some correlation between features (realistic user patterns)
        for i in range(1, n_features):
            queries[:, i] = 0.7 * queries[:, i-1] + 0.3 * queries[:, i]
        
        # Create DataFrame
        df = pd.DataFrame(queries, columns=[f"feature_{i}" for i in range(n_features)])
        df["user_id"] = user_id
        df["query_timestamp"] = pd.date_range(
            start="2024-01-01", periods=n_queries, freq="1min"
        )
        df["query_type"] = "legitimate"
        df["is_stealing"] = False
        
        return df
    
    def generate_stealing_queries(
        self,
        n_queries: int = 1000,
        n_features: int = 10,
        user_id: str = "suspicious_user",
        strategy: str = "exhaustive"
    ) -> pd.DataFrame:
        """
        Generate model stealing queries with different attack strategies.
        
        Args:
            n_queries: Number of queries to generate
            n_features: Number of features in each query
            user_id: User identifier
            strategy: Attack strategy ('exhaustive', 'adversarial', 'systematic')
            
        Returns:
            pd.DataFrame: Generated queries with metadata
        """
        if strategy == "exhaustive":
            # Exhaustive exploration of input space
            queries = np.random.uniform(low=-5, high=5, size=(n_queries, n_features))
        elif strategy == "adversarial":
            # Adversarial examples with small perturbations
            base_queries = np.random.normal(loc=0, scale=1, size=(n_queries, n_features))
            noise = np.random.normal(loc=0, scale=0.1, size=(n_queries, n_features))
            queries = base_queries + noise
        elif strategy == "systematic":
            # Systematic grid search
            grid_size = int(np.ceil(n_queries ** (1/n_features)))
            grid_points = np.linspace(-3, 3, grid_size)
            queries = np.array(np.meshgrid(*[grid_points] * n_features)).T.reshape(-1, n_features)
            queries = queries[:n_queries]  # Truncate if needed
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Create DataFrame
        df = pd.DataFrame(queries, columns=[f"feature_{i}" for i in range(n_features)])
        df["user_id"] = user_id
        df["query_timestamp"] = pd.date_range(
            start="2024-01-01", periods=len(df), freq="30s"  # Faster queries
        )
        df["query_type"] = f"stealing_{strategy}"
        df["is_stealing"] = True
        
        return df
    
    def generate_mixed_dataset(
        self,
        n_legitimate: int = 1000,
        n_stealing: int = 200,
        n_features: int = 10,
        strategies: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Generate a mixed dataset with both legitimate and stealing queries.
        
        Args:
            n_legitimate: Number of legitimate queries
            n_stealing: Number of stealing queries
            n_features: Number of features
            strategies: List of stealing strategies to include
            
        Returns:
            pd.DataFrame: Combined dataset
        """
        if strategies is None:
            strategies = ["exhaustive", "adversarial", "systematic"]
        
        # Generate legitimate queries
        legitimate_df = self.generate_legitimate_queries(
            n_queries=n_legitimate,
            n_features=n_features,
            user_id="legitimate_user"
        )
        
        # Generate stealing queries for each strategy
        stealing_dfs = []
        n_per_strategy = n_stealing // len(strategies)
        
        for i, strategy in enumerate(strategies):
            n_queries = n_per_strategy if i < len(strategies) - 1 else n_stealing - i * n_per_strategy
            stealing_df = self.generate_stealing_queries(
                n_queries=n_queries,
                n_features=n_features,
                user_id=f"suspicious_user_{strategy}",
                strategy=strategy
            )
            stealing_dfs.append(stealing_df)
        
        # Combine all datasets
        all_dfs = [legitimate_df] + stealing_dfs
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        # Shuffle the dataset
        combined_df = combined_df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        
        return combined_df
    
    def add_query_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add additional metadata to queries for analysis.
        
        Args:
            df: Input DataFrame with queries
            
        Returns:
            pd.DataFrame: DataFrame with added metadata
        """
        df = df.copy()
        
        # Add query hash for uniqueness
        df["query_hash"] = df.apply(
            lambda row: hashlib.md5(
                str(row[[col for col in df.columns if col.startswith("feature_")]].values).encode()
            ).hexdigest()[:8],
            axis=1
        )
        
        # Add query complexity (entropy)
        feature_cols = [col for col in df.columns if col.startswith("feature_")]
        df["query_entropy"] = df[feature_cols].apply(
            lambda row: entropy(np.histogram(row, bins=10)[0] + 1e-9), axis=1
        )
        
        # Add query magnitude
        df["query_magnitude"] = df[feature_cols].apply(
            lambda row: np.linalg.norm(row), axis=1
        )
        
        # Add query diversity (standard deviation)
        df["query_diversity"] = df[feature_cols].apply(
            lambda row: np.std(row), axis=1
        )
        
        return df
    
    def generate_user_behavior_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate user behavior features for analysis.
        
        Args:
            df: Input DataFrame with queries
            
        Returns:
            pd.DataFrame: DataFrame with user behavior features
        """
        df = df.copy()
        
        # Group by user and calculate behavior features
        user_features = []
        
        for user_id in df["user_id"].unique():
            user_df = df[df["user_id"] == user_id].copy()
            user_df = user_df.sort_values("query_timestamp")
            
            # Calculate time-based features
            user_df["time_since_last_query"] = user_df["query_timestamp"].diff().dt.total_seconds()
            user_df["query_frequency"] = 1 / user_df["time_since_last_query"].fillna(1)
            
            # Calculate query pattern features
            feature_cols = [col for col in df.columns if col.startswith("feature_")]
            user_df["avg_query_entropy"] = user_df["query_entropy"].rolling(window=10).mean()
            user_df["avg_query_magnitude"] = user_df["query_magnitude"].rolling(window=10).mean()
            user_df["avg_query_diversity"] = user_df["query_diversity"].rolling(window=10).mean()
            
            # Calculate query similarity (cosine similarity with previous queries)
            if len(user_df) > 1:
                similarities = []
                for i in range(1, len(user_df)):
                    prev_query = user_df.iloc[i-1][feature_cols].values
                    curr_query = user_df.iloc[i][feature_cols].values
                    similarity = np.dot(prev_query, curr_query) / (
                        np.linalg.norm(prev_query) * np.linalg.norm(curr_query) + 1e-9
                    )
                    similarities.append(similarity)
                user_df["query_similarity"] = [0] + similarities
            else:
                user_df["query_similarity"] = 0
            
            user_features.append(user_df)
        
        return pd.concat(user_features, ignore_index=True)
