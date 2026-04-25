"""Query feature engineering for model stealing detection."""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler


class QueryFeatureExtractor:
    """
    Extract features from user queries for model stealing detection.
    
    This class implements various feature extraction methods to identify
    suspicious query patterns that may indicate model stealing attempts.
    """
    
    def __init__(self, n_bins: int = 10, window_size: int = 10):
        """
        Initialize the feature extractor.
        
        Args:
            n_bins: Number of bins for histogram-based features
            window_size: Window size for rolling features
        """
        self.n_bins = n_bins
        self.window_size = window_size
        self.scaler = StandardScaler()
        
    def extract_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract basic statistical features from queries.
        
        Args:
            df: DataFrame with query data
            
        Returns:
            pd.DataFrame: DataFrame with basic features
        """
        df = df.copy()
        feature_cols = [col for col in df.columns if col.startswith("feature_")]
        
        # Basic statistical features
        df["query_mean"] = df[feature_cols].mean(axis=1)
        df["query_std"] = df[feature_cols].std(axis=1)
        df["query_min"] = df[feature_cols].min(axis=1)
        df["query_max"] = df[feature_cols].max(axis=1)
        df["query_range"] = df["query_max"] - df["query_min"]
        df["query_median"] = df[feature_cols].median(axis=1)
        df["query_q25"] = df[feature_cols].quantile(0.25, axis=1)
        df["query_q75"] = df[feature_cols].quantile(0.75, axis=1)
        df["query_iqr"] = df["query_q75"] - df["query_q25"]
        
        # Norm and distance features
        df["query_l2_norm"] = np.linalg.norm(df[feature_cols], axis=1)
        df["query_l1_norm"] = np.sum(np.abs(df[feature_cols]), axis=1)
        df["query_l_inf_norm"] = np.max(np.abs(df[feature_cols]), axis=1)
        
        # Entropy and diversity features
        df["query_entropy"] = df[feature_cols].apply(
            lambda row: entropy(np.histogram(row, bins=self.n_bins)[0] + 1e-9), axis=1
        )
        df["query_diversity"] = df[feature_cols].std(axis=1)
        
        return df
    
    def extract_distribution_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract distribution-based features from queries.
        
        Args:
            df: DataFrame with query data
            
        Returns:
            pd.DataFrame: DataFrame with distribution features
        """
        df = df.copy()
        feature_cols = [col for col in df.columns if col.startswith("feature_")]
        
        # Skewness and kurtosis
        df["query_skewness"] = df[feature_cols].apply(
            lambda row: stats.skew(row), axis=1
        )
        df["query_kurtosis"] = df[feature_cols].apply(
            lambda row: stats.kurtosis(row), axis=1
        )
        
        # Normality test (Shapiro-Wilk test statistic)
        df["query_normality"] = df[feature_cols].apply(
            lambda row: stats.shapiro(row)[0] if len(row) >= 3 else 0, axis=1
        )
        
        # Outlier detection (Z-score based)
        df["query_outlier_count"] = df[feature_cols].apply(
            lambda row: np.sum(np.abs(stats.zscore(row)) > 2), axis=1
        )
        
        # Percentile features
        for p in [10, 20, 30, 40, 60, 70, 80, 90]:
            df[f"query_p{p}"] = df[feature_cols].quantile(p/100, axis=1)
        
        return df
    
    def extract_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract temporal features from query sequences.
        
        Args:
            df: DataFrame with query data and timestamps
            
        Returns:
            pd.DataFrame: DataFrame with temporal features
        """
        df = df.copy()
        
        # Sort by user and timestamp
        df = df.sort_values(["user_id", "query_timestamp"])
        
        # Time-based features
        df["time_since_last_query"] = df.groupby("user_id")["query_timestamp"].diff().dt.total_seconds()
        df["query_frequency"] = 1 / df["time_since_last_query"].fillna(1)
        df["query_frequency"] = df["query_frequency"].replace([np.inf, -np.inf], 0)
        
        # Rolling window features
        feature_cols = [col for col in df.columns if col.startswith("feature_")]
        
        for col in feature_cols:
            df[f"{col}_rolling_mean"] = df.groupby("user_id")[col].rolling(
                window=self.window_size, min_periods=1
            ).mean().reset_index(0, drop=True)
            df[f"{col}_rolling_std"] = df.groupby("user_id")[col].rolling(
                window=self.window_size, min_periods=1
            ).std().reset_index(0, drop=True)
        
        # Query similarity with previous queries
        df["query_similarity"] = 0.0
        for user_id in df["user_id"].unique():
            user_mask = df["user_id"] == user_id
            user_df = df[user_mask].copy()
            
            if len(user_df) > 1:
                similarities = []
                for i in range(1, len(user_df)):
                    prev_query = user_df.iloc[i-1][feature_cols].values
                    curr_query = user_df.iloc[i][feature_cols].values
                    similarity = np.dot(prev_query, curr_query) / (
                        np.linalg.norm(prev_query) * np.linalg.norm(curr_query) + 1e-9
                    )
                    similarities.append(similarity)
                
                df.loc[user_mask, "query_similarity"] = [0] + similarities
        
        return df
    
    def extract_anomaly_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract anomaly detection features from queries.
        
        Args:
            df: DataFrame with query data
            
        Returns:
            pd.DataFrame: DataFrame with anomaly features
        """
        df = df.copy()
        feature_cols = [col for col in df.columns if col.startswith("feature_")]
        
        # Isolation Forest features (simplified)
        df["query_isolation_score"] = df[feature_cols].apply(
            lambda row: self._isolation_score(row), axis=1
        )
        
        # Local Outlier Factor (simplified)
        df["query_lof_score"] = df[feature_cols].apply(
            lambda row: self._lof_score(row), axis=1
        )
        
        # One-Class SVM features (simplified)
        df["query_ocsvm_score"] = df[feature_cols].apply(
            lambda row: self._ocsvm_score(row), axis=1
        )
        
        return df
    
    def _isolation_score(self, query: np.ndarray) -> float:
        """Calculate simplified isolation score for a query."""
        # Simplified isolation score based on distance from center
        center = np.mean(query)
        distances = np.abs(query - center)
        return np.mean(distances)
    
    def _lof_score(self, query: np.ndarray) -> float:
        """Calculate simplified LOF score for a query."""
        # Simplified LOF based on local density
        distances = np.abs(query - np.mean(query))
        local_density = 1 / (np.mean(distances) + 1e-9)
        return local_density
    
    def _ocsvm_score(self, query: np.ndarray) -> float:
        """Calculate simplified OCSVM score for a query."""
        # Simplified OCSVM based on distance from decision boundary
        center = np.mean(query)
        distance = np.linalg.norm(query - center)
        return distance
    
    def extract_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all features from queries.
        
        Args:
            df: DataFrame with query data
            
        Returns:
            pd.DataFrame: DataFrame with all extracted features
        """
        # Extract different types of features
        df = self.extract_basic_features(df)
        df = self.extract_distribution_features(df)
        df = self.extract_temporal_features(df)
        df = self.extract_anomaly_features(df)
        
        # Remove original feature columns to avoid duplication
        feature_cols = [col for col in df.columns if col.startswith("feature_")]
        df = df.drop(columns=feature_cols)
        
        return df
    
    def fit_scaler(self, df: pd.DataFrame) -> None:
        """
        Fit the scaler on training data.
        
        Args:
            df: Training DataFrame
        """
        feature_cols = [col for col in df.columns if col.startswith("query_")]
        self.scaler.fit(df[feature_cols])
    
    def transform_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using fitted scaler.
        
        Args:
            df: DataFrame to transform
            
        Returns:
            pd.DataFrame: Transformed DataFrame
        """
        df = df.copy()
        feature_cols = [col for col in df.columns if col.startswith("query_")]
        
        if len(feature_cols) > 0:
            df[feature_cols] = self.scaler.transform(df[feature_cols])
        
        return df
