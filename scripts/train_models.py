#!/usr/bin/env python3
"""Training script for model stealing detection models."""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from data.synthetic import ModelStealingDataGenerator
from features.query_features import QueryFeatureExtractor
from models.detection_models import (
    EntropyBasedDetector,
    IsolationForestDetector,
    OneClassSVMDetector,
    RandomForestDetector,
    NeuralNetworkDetector,
    EnsembleDetector
)
from eval.metrics import ModelStealingLeaderboard
from viz.explainability import ModelStealingExplainer, ModelStealingVisualizer
from utils.device import get_device, set_deterministic
from utils.logging import setup_logging


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def generate_data(config: dict) -> pd.DataFrame:
    """Generate synthetic data for training."""
    logger = setup_logging()
    logger.info("Generating synthetic data...")
    
    generator = ModelStealingDataGenerator(seed=config['data']['seed'])
    df = generator.generate_mixed_dataset(
        n_legitimate=config['data']['n_legitimate'],
        n_stealing=config['data']['n_stealing'],
        n_features=config['data']['n_features'],
        strategies=config['data']['strategies']
    )
    
    # Add metadata and behavior features
    df = generator.add_query_metadata(df)
    df = generator.generate_user_behavior_features(df)
    
    logger.info(f"Generated {len(df)} queries with {len(df[df['is_stealing']])} stealing attempts")
    return df


def extract_features(df: pd.DataFrame, config: dict) -> tuple:
    """Extract features from the dataset."""
    logger = setup_logging()
    logger.info("Extracting features...")
    
    feature_extractor = QueryFeatureExtractor(
        n_bins=config['features']['n_bins'],
        window_size=config['features']['window_size']
    )
    
    # Extract all features
    df_features = feature_extractor.extract_all_features(df)
    
    # Prepare training data
    feature_cols = [col for col in df_features.columns if col.startswith("query_")]
    X = df_features[feature_cols].values
    y = df_features['is_stealing'].values
    
    # Scale features if requested
    if config['features']['scale_features']:
        feature_extractor.fit_scaler(df_features)
        X = feature_extractor.transform_features(df_features)[feature_cols].values
    
    logger.info(f"Extracted {X.shape[1]} features from {X.shape[0]} samples")
    return X, y, feature_cols, feature_extractor


def train_models(X: np.ndarray, y: np.ndarray, feature_cols: list, config: dict) -> dict:
    """Train all detection models."""
    logger = setup_logging()
    logger.info("Training detection models...")
    
    models = {}
    
    # Entropy-based detector
    logger.info("Training entropy-based detector...")
    entropy_detector = EntropyBasedDetector(
        threshold=config['models']['entropy_detector']['threshold'],
        n_bins=config['models']['entropy_detector']['n_bins']
    )
    entropy_detector.fit(X)
    models['EntropyBasedDetector'] = entropy_detector
    
    # Isolation Forest
    logger.info("Training isolation forest...")
    isolation_forest = IsolationForestDetector(
        contamination=config['models']['isolation_forest']['contamination'],
        random_state=config['models']['isolation_forest']['random_state']
    )
    isolation_forest.fit(X)
    models['IsolationForest'] = isolation_forest
    
    # One-Class SVM
    logger.info("Training one-class SVM...")
    one_class_svm = OneClassSVMDetector(
        nu=config['models']['one_class_svm']['nu'],
        kernel=config['models']['one_class_svm']['kernel'],
        gamma=config['models']['one_class_svm']['gamma']
    )
    one_class_svm.fit(X)
    models['OneClassSVM'] = one_class_svm
    
    # Random Forest
    logger.info("Training random forest...")
    random_forest = RandomForestDetector(
        n_estimators=config['models']['random_forest']['n_estimators'],
        random_state=config['models']['random_forest']['random_state']
    )
    random_forest.fit(X, y)
    models['RandomForest'] = random_forest
    
    # Neural Network
    logger.info("Training neural network...")
    device = get_device()
    neural_network = NeuralNetworkDetector(
        input_dim=X.shape[1],
        hidden_dims=config['models']['neural_network']['hidden_dims'],
        dropout_rate=config['models']['neural_network']['dropout_rate'],
        learning_rate=config['models']['neural_network']['learning_rate'],
        device=device
    )
    neural_network.fit(
        X, y,
        epochs=config['models']['neural_network']['epochs'],
        batch_size=config['models']['neural_network']['batch_size']
    )
    models['NeuralNetwork'] = neural_network
    
    # Ensemble
    logger.info("Training ensemble...")
    ensemble = EnsembleDetector(
        models=[models['RandomForest'], models['NeuralNetwork']],
        weights=[0.5, 0.5]
    )
    ensemble.fit(X, y)
    models['Ensemble'] = ensemble
    
    logger.info(f"Trained {len(models)} models successfully")
    return models


def evaluate_models(models: dict, X: np.ndarray, y: np.ndarray, feature_cols: list) -> pd.DataFrame:
    """Evaluate all trained models."""
    logger = setup_logging()
    logger.info("Evaluating models...")
    
    leaderboard = ModelStealingLeaderboard()
    
    for model_name, model in models.items():
        logger.info(f"Evaluating {model_name}...")
        
        # Make predictions
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        # Add to leaderboard
        leaderboard.add_model_result(model_name, y, predictions, probabilities)
    
    # Get leaderboard
    leaderboard_df = leaderboard.get_leaderboard(sort_by="f1_score")
    
    logger.info("Model evaluation completed")
    return leaderboard_df


def create_visualizations(models: dict, X: np.ndarray, y: np.ndarray, feature_cols: list, output_dir: str):
    """Create visualization plots."""
    logger = setup_logging()
    logger.info("Creating visualizations...")
    
    visualizer = ModelStealingVisualizer()
    plots = {}
    
    # Get best model for detailed analysis
    best_model_name = max(models.keys(), key=lambda k: models[k].score(X, y))
    best_model = models[best_model_name]
    
    # Make predictions
    predictions = best_model.predict(X)
    probabilities = best_model.predict_proba(X)
    
    # ROC curve
    plots['roc_curve'] = visualizer.plot_roc_curve(
        y, probabilities[:, 1], 
        title=f"ROC Curve - {best_model_name}"
    )
    
    # Precision-Recall curve
    plots['pr_curve'] = visualizer.plot_precision_recall_curve(
        y, probabilities[:, 1],
        title=f"Precision-Recall Curve - {best_model_name}"
    )
    
    # Confusion matrix
    plots['confusion_matrix'] = visualizer.plot_confusion_matrix(
        y, predictions,
        title=f"Confusion Matrix - {best_model_name}"
    )
    
    # Prediction distribution
    plots['prediction_distribution'] = visualizer.plot_prediction_distribution(
        probabilities[:, 1], y,
        title=f"Prediction Distribution - {best_model_name}"
    )
    
    # Feature importance (if available)
    if hasattr(best_model, 'feature_importances_'):
        importance_scores = {}
        for i, feature_name in enumerate(feature_cols):
            importance_scores[feature_name] = best_model.feature_importances_[i]
        
        plots['feature_importance'] = visualizer.plot_feature_importance(
            importance_scores,
            title=f"Feature Importance - {best_model_name}"
        )
    
    # Save plots
    visualizer.save_plots(plots, output_dir)
    logger.info(f"Visualizations saved to {output_dir}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train model stealing detection models")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Configuration file path")
    parser.add_argument("--output-dir", type=str, default="assets", help="Output directory for results")
    parser.add_argument("--save-models", action="store_true", help="Save trained models")
    parser.add_argument("--create-plots", action="store_true", help="Create visualization plots")
    
    args = parser.parse_args()
    
    # Set deterministic behavior
    set_deterministic(42)
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting model training pipeline...")
    
    # Load configuration
    config = load_config(args.config)
    logger.info(f"Loaded configuration from {args.config}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Generate data
        df = generate_data(config)
        
        # Extract features
        X, y, feature_cols, feature_extractor = extract_features(df, config)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config['evaluation']['test_size'],
            random_state=config['evaluation']['random_state'],
            stratify=y
        )
        
        # Train models
        models = train_models(X_train, y_train, feature_cols, config)
        
        # Evaluate models
        leaderboard_df = evaluate_models(models, X_test, y_test, feature_cols)
        
        # Save results
        leaderboard_df.to_csv(output_dir / "leaderboard.csv", index=False)
        logger.info(f"Leaderboard saved to {output_dir / 'leaderboard.csv'}")
        
        # Create visualizations
        if args.create_plots:
            plots_dir = output_dir / "plots"
            plots_dir.mkdir(exist_ok=True)
            create_visualizations(models, X_test, y_test, feature_cols, str(plots_dir))
        
        # Save models
        if args.save_models:
            import joblib
            models_dir = output_dir / "models"
            models_dir.mkdir(exist_ok=True)
            
            for model_name, model in models.items():
                model_path = models_dir / f"{model_name.lower()}.joblib"
                joblib.dump(model, model_path)
                logger.info(f"Model saved to {model_path}")
        
        # Print results
        print("\n" + "="*50)
        print("TRAINING RESULTS")
        print("="*50)
        print(leaderboard_df.to_string(index=False))
        
        best_model = leaderboard_df.iloc[0]
        print(f"\nBest Model: {best_model['model_name']}")
        print(f"F1 Score: {best_model['f1_score']:.3f}")
        print(f"ROC AUC: {best_model['roc_auc']:.3f}")
        
        logger.info("Training pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
