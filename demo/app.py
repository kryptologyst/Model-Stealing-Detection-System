"""Streamlit demo for model stealing detection."""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data.synthetic import ModelStealingDataGenerator
from features.query_features import QueryFeatureExtractor
from models.detection_models import (
    EntropyBasedDetector,
    IsolationForestDetector,
    OneClassSVMDetector,
    RandomForestDetector,
    EnsembleDetector
)
from eval.metrics import ModelStealingMetrics, ModelStealingLeaderboard
from viz.explainability import ModelStealingExplainer, ModelStealingVisualizer
from defenses.api_defenses import APIDefenseSystem, RateLimiter, QueryDeduplicator, ResponseRandomizer
from utils.device import get_device, set_deterministic
from utils.logging import setup_logging

# Set page config
st.set_page_config(
    page_title="Model Stealing Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set deterministic behavior
set_deterministic(42)

# Setup logging
logger = setup_logging()

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🛡️ Model Stealing Detection System</h1>', unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("""
    <div class="warning-box">
    <strong>⚠️ Disclaimer:</strong> This is a research and educational demonstration of model stealing detection techniques. 
    This system is designed for defensive purposes only and should not be used for offensive activities or exploitation.
    The models and data shown are synthetic and for demonstration purposes only.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Configuration")
    
    # Data generation parameters
    st.sidebar.header("Data Generation")
    n_legitimate = st.sidebar.slider("Legitimate Queries", 100, 2000, 1000)
    n_stealing = st.sidebar.slider("Stealing Queries", 50, 500, 200)
    n_features = st.sidebar.slider("Number of Features", 5, 20, 10)
    
    # Model parameters
    st.sidebar.header("Model Parameters")
    entropy_threshold = st.sidebar.slider("Entropy Threshold", 1.0, 5.0, 2.0)
    contamination = st.sidebar.slider("Contamination Rate", 0.01, 0.5, 0.1)
    
    # Defense parameters
    st.sidebar.header("Defense Parameters")
    max_requests_per_minute = st.sidebar.slider("Max Requests/Minute", 10, 100, 60)
    max_duplicates = st.sidebar.slider("Max Duplicates", 1, 20, 5)
    noise_level = st.sidebar.slider("Response Noise Level", 0.0, 0.5, 0.1)
    
    # Main content
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Overview", 
        "🤖 Model Training", 
        "🔍 Detection Results", 
        "🛡️ Defense System", 
        "📈 Analysis & Insights"
    ])
    
    with tab1:
        show_data_overview(n_legitimate, n_stealing, n_features)
    
    with tab2:
        show_model_training(n_legitimate, n_stealing, n_features, entropy_threshold, contamination)
    
    with tab3:
        show_detection_results(n_legitimate, n_stealing, n_features, entropy_threshold, contamination)
    
    with tab4:
        show_defense_system(max_requests_per_minute, max_duplicates, noise_level)
    
    with tab5:
        show_analysis_insights(n_legitimate, n_stealing, n_features, entropy_threshold, contamination)


def show_data_overview(n_legitimate: int, n_stealing: int, n_features: int):
    """Show data overview tab."""
    st.header("📊 Data Overview")
    
    # Generate data
    with st.spinner("Generating synthetic data..."):
        generator = ModelStealingDataGenerator(seed=42)
        df = generator.generate_mixed_dataset(
            n_legitimate=n_legitimate,
            n_stealing=n_stealing,
            n_features=n_features
        )
        df = generator.add_query_metadata(df)
        df = generator.generate_user_behavior_features(df)
    
    # Data summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Queries", len(df))
    
    with col2:
        st.metric("Legitimate Queries", len(df[df['is_stealing'] == False]))
    
    with col3:
        st.metric("Stealing Queries", len(df[df['is_stealing'] == True]))
    
    with col4:
        st.metric("Stealing Rate", f"{len(df[df['is_stealing'] == True]) / len(df) * 100:.1f}%")
    
    # Data visualization
    st.subheader("Query Distribution")
    
    # Feature distribution
    feature_cols = [col for col in df.columns if col.startswith("feature_")]
    
    if feature_cols:
        # Sample features for visualization
        sample_features = feature_cols[:min(4, len(feature_cols))]
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=sample_features,
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        for i, feature in enumerate(sample_features):
            row = i // 2 + 1
            col = i % 2 + 1
            
            # Plot distributions for each class
            normal_data = df[df['is_stealing'] == False][feature]
            stealing_data = df[df['is_stealing'] == True][feature]
            
            fig.add_trace(
                go.Histogram(x=normal_data, name='Normal', opacity=0.7, nbinsx=20),
                row=row, col=col
            )
            fig.add_trace(
                go.Histogram(x=stealing_data, name='Stealing', opacity=0.7, nbinsx=20),
                row=row, col=col
            )
        
        fig.update_layout(height=600, showlegend=False, title_text="Feature Distributions by Class")
        st.plotly_chart(fig, use_container_width=True)
    
    # Query patterns
    st.subheader("Query Patterns")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Entropy distribution
        fig = px.histogram(
            df, x='query_entropy', color='is_stealing',
            title='Query Entropy Distribution',
            labels={'query_entropy': 'Query Entropy', 'count': 'Count'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Magnitude distribution
        fig = px.histogram(
            df, x='query_magnitude', color='is_stealing',
            title='Query Magnitude Distribution',
            labels={'query_magnitude': 'Query Magnitude', 'count': 'Count'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # User behavior
    st.subheader("User Behavior Analysis")
    
    user_stats = df.groupby('user_id').agg({
        'is_stealing': 'first',
        'query_entropy': 'mean',
        'query_magnitude': 'mean',
        'query_frequency': 'mean',
        'query_similarity': 'mean'
    }).reset_index()
    
    # User behavior scatter plot
    fig = px.scatter(
        user_stats, x='query_entropy', y='query_magnitude',
        color='is_stealing', size='query_frequency',
        title='User Behavior Patterns',
        labels={'query_entropy': 'Average Query Entropy', 'query_magnitude': 'Average Query Magnitude'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Data table
    st.subheader("Sample Data")
    st.dataframe(df.head(100), use_container_width=True)


def show_model_training(n_legitimate: int, n_stealing: int, n_features: int, entropy_threshold: float, contamination: float):
    """Show model training tab."""
    st.header("🤖 Model Training")
    
    # Generate data
    with st.spinner("Generating training data..."):
        generator = ModelStealingDataGenerator(seed=42)
        df = generator.generate_mixed_dataset(
            n_legitimate=n_legitimate,
            n_stealing=n_stealing,
            n_features=n_features
        )
        df = generator.add_query_metadata(df)
        df = generator.generate_user_behavior_features(df)
    
    # Feature extraction
    with st.spinner("Extracting features..."):
        feature_extractor = QueryFeatureExtractor()
        df_features = feature_extractor.extract_all_features(df)
        
        # Prepare training data
        feature_cols = [col for col in df_features.columns if col.startswith("query_")]
        X = df_features[feature_cols].values
        y = df_features['is_stealing'].values
        
        # Fit scaler
        feature_extractor.fit_scaler(df_features)
        X_scaled = feature_extractor.transform_features(df_features)[feature_cols].values
    
    # Model selection
    st.subheader("Model Selection")
    
    models = {
        "Entropy-Based Detector": EntropyBasedDetector(threshold=entropy_threshold),
        "Isolation Forest": IsolationForestDetector(contamination=contamination),
        "One-Class SVM": OneClassSVMDetector(nu=contamination),
        "Random Forest": RandomForestDetector(n_estimators=100),
    }
    
    selected_models = st.multiselect(
        "Select models to train:",
        list(models.keys()),
        default=list(models.keys())
    )
    
    if st.button("Train Models"):
        with st.spinner("Training models..."):
            # Train selected models
            trained_models = {}
            predictions = {}
            probabilities = {}
            
            for model_name in selected_models:
                model = models[model_name]
                
                try:
                    if model_name == "EntropyBasedDetector":
                        # Use original features for entropy-based detector
                        model.fit(X)
                        pred = model.predict(X)
                        prob = model.predict_proba(X)
                    else:
                        # Use scaled features for other models
                        model.fit(X_scaled, y)
                        pred = model.predict(X_scaled)
                        prob = model.predict_proba(X_scaled)
                    
                    trained_models[model_name] = model
                    predictions[model_name] = pred
                    probabilities[model_name] = prob
                    
                    st.success(f"✅ {model_name} trained successfully")
                    
                except Exception as e:
                    st.error(f"❌ {model_name} training failed: {str(e)}")
            
            # Store in session state
            st.session_state.trained_models = trained_models
            st.session_state.predictions = predictions
            st.session_state.probabilities = probabilities
            st.session_state.X = X
            st.session_state.X_scaled = X_scaled
            st.session_state.y = y
            st.session_state.feature_names = feature_cols
    
    # Model performance
    if 'trained_models' in st.session_state:
        st.subheader("Model Performance")
        
        # Calculate metrics
        metrics_calculator = ModelStealingMetrics()
        leaderboard = ModelStealingLeaderboard()
        
        for model_name, pred in st.session_state.predictions.items():
            prob = st.session_state.probabilities[model_name]
            metrics = metrics_calculator.calculate_all_metrics(st.session_state.y, pred, prob)
            leaderboard.add_model_result(model_name, st.session_state.y, pred, prob)
        
        # Display leaderboard
        leaderboard_df = leaderboard.get_leaderboard(sort_by="f1_score")
        st.dataframe(leaderboard_df, use_container_width=True)
        
        # Performance visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # F1 Score comparison
            fig = px.bar(
                leaderboard_df, x='model_name', y='f1_score',
                title='F1 Score Comparison',
                labels={'model_name': 'Model', 'f1_score': 'F1 Score'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # ROC AUC comparison
            if 'roc_auc' in leaderboard_df.columns:
                fig = px.bar(
                    leaderboard_df, x='model_name', y='roc_auc',
                    title='ROC AUC Comparison',
                    labels={'model_name': 'Model', 'roc_auc': 'ROC AUC'}
                )
                st.plotly_chart(fig, use_container_width=True)


def show_detection_results(n_legitimate: int, n_stealing: int, n_features: int, entropy_threshold: float, contamination: float):
    """Show detection results tab."""
    st.header("🔍 Detection Results")
    
    if 'trained_models' not in st.session_state:
        st.warning("Please train models first in the 'Model Training' tab.")
        return
    
    # Model selection for analysis
    selected_model = st.selectbox(
        "Select model for analysis:",
        list(st.session_state.trained_models.keys())
    )
    
    if selected_model:
        model = st.session_state.trained_models[selected_model]
        predictions = st.session_state.predictions[selected_model]
        probabilities = st.session_state.probabilities[selected_model]
        
        # Confusion matrix
        st.subheader("Confusion Matrix")
        
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(st.session_state.y, predictions)
        
        fig = px.imshow(
            cm, text_auto=True, aspect="auto",
            title=f"Confusion Matrix - {selected_model}",
            labels=dict(x="Predicted", y="Actual", color="Count")
        )
        fig.update_xaxes(tickvals=[0, 1], ticktext=["Normal", "Stealing"])
        fig.update_yaxes(tickvals=[0, 1], ticktext=["Normal", "Stealing"])
        st.plotly_chart(fig, use_container_width=True)
        
        # ROC and PR curves
        col1, col2 = st.columns(2)
        
        with col1:
            # ROC curve
            from sklearn.metrics import roc_curve, auc
            fpr, tpr, _ = roc_curve(st.session_state.y, probabilities[:, 1])
            roc_auc = auc(fpr, tpr)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC = {roc_auc:.2f})'))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')))
            fig.update_layout(title='ROC Curve', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Precision-Recall curve
            from sklearn.metrics import precision_recall_curve, average_precision_score
            precision, recall, _ = precision_recall_curve(st.session_state.y, probabilities[:, 1])
            pr_auc = average_precision_score(st.session_state.y, probabilities[:, 1])
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name=f'PR (AUC = {pr_auc:.2f})'))
            fig.update_layout(title='Precision-Recall Curve', xaxis_title='Recall', yaxis_title='Precision')
            st.plotly_chart(fig, use_container_width=True)
        
        # Prediction distribution
        st.subheader("Prediction Distribution")
        
        fig = px.histogram(
            x=probabilities[:, 1], color=st.session_state.y.astype(str),
            title='Prediction Score Distribution',
            labels={'x': 'Prediction Score', 'count': 'Count'},
            nbins=30
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance (if available)
        if hasattr(model, 'feature_importances_'):
            st.subheader("Feature Importance")
            
            importance_df = pd.DataFrame({
                'feature': st.session_state.feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=True)
            
            fig = px.bar(
                importance_df, x='importance', y='feature',
                title='Feature Importance',
                labels={'importance': 'Importance', 'feature': 'Feature'}
            )
            st.plotly_chart(fig, use_container_width=True)


def show_defense_system(max_requests_per_minute: int, max_duplicates: int, noise_level: float):
    """Show defense system tab."""
    st.header("🛡️ Defense System")
    
    # Initialize defense system
    rate_limiter = RateLimiter(max_requests_per_minute=max_requests_per_minute)
    query_deduplicator = QueryDeduplicator(max_duplicates=max_duplicates)
    response_randomizer = ResponseRandomizer(noise_level=noise_level)
    
    defense_system = APIDefenseSystem(
        rate_limiter=rate_limiter,
        query_deduplicator=query_deduplicator,
        response_randomizer=response_randomizer
    )
    
    # Simulate API requests
    st.subheader("API Request Simulation")
    
    if st.button("Simulate API Requests"):
        with st.spinner("Simulating API requests..."):
            # Generate test queries
            generator = ModelStealingDataGenerator(seed=42)
            test_queries = generator.generate_mixed_dataset(
                n_legitimate=50,
                n_stealing=20,
                n_features=10
            )
            
            # Simulate API calls
            results = []
            for idx, row in test_queries.iterrows():
                user_id = row['user_id']
                query = row[[col for col in row.index if col.startswith('feature_')]].values
                model_response = np.random.random(5)  # Simulate model response
                
                is_allowed, response, reason = defense_system.process_query(user_id, query, model_response)
                
                results.append({
                    'user_id': user_id,
                    'query_id': idx,
                    'is_allowed': is_allowed,
                    'reason': reason,
                    'is_stealing': row['is_stealing']
                })
            
            # Store results
            st.session_state.defense_results = pd.DataFrame(results)
    
    # Display results
    if 'defense_results' in st.session_state:
        results_df = st.session_state.defense_results
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Requests", len(results_df))
        
        with col2:
            st.metric("Allowed Requests", len(results_df[results_df['is_allowed'] == True]))
        
        with col3:
            st.metric("Blocked Requests", len(results_df[results_df['is_allowed'] == False]))
        
        with col4:
            st.metric("Block Rate", f"{len(results_df[results_df['is_allowed'] == False]) / len(results_df) * 100:.1f}%")
        
        # Defense effectiveness
        st.subheader("Defense Effectiveness")
        
        # Block rate by user type
        block_rate_by_type = results_df.groupby('is_stealing')['is_allowed'].apply(
            lambda x: (x == False).sum() / len(x) * 100
        ).reset_index()
        block_rate_by_type['user_type'] = block_rate_by_type['is_stealing'].map({False: 'Legitimate', True: 'Stealing'})
        
        fig = px.bar(
            block_rate_by_type, x='user_type', y='is_allowed',
            title='Block Rate by User Type',
            labels={'user_type': 'User Type', 'is_allowed': 'Block Rate (%)'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Defense actions
        st.subheader("Defense Actions")
        
        action_counts = results_df['reason'].value_counts()
        
        fig = px.pie(
            values=action_counts.values, names=action_counts.index,
            title='Distribution of Defense Actions'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed results
        st.subheader("Detailed Results")
        st.dataframe(results_df, use_container_width=True)
        
        # Suspicious users
        suspicious_users = defense_system.get_suspicious_users()
        if suspicious_users:
            st.subheader("Suspicious Users")
            st.write("Users flagged as suspicious:", suspicious_users)


def show_analysis_insights(n_legitimate: int, n_stealing: int, n_features: int, entropy_threshold: float, contamination: float):
    """Show analysis and insights tab."""
    st.header("📈 Analysis & Insights")
    
    if 'trained_models' not in st.session_state:
        st.warning("Please train models first in the 'Model Training' tab.")
        return
    
    # Model comparison
    st.subheader("Model Comparison")
    
    # Calculate comprehensive metrics
    metrics_calculator = ModelStealingMetrics()
    leaderboard = ModelStealingLeaderboard()
    
    for model_name, pred in st.session_state.predictions.items():
        prob = st.session_state.probabilities[model_name]
        metrics = metrics_calculator.calculate_all_metrics(st.session_state.y, pred, prob)
        leaderboard.add_model_result(model_name, st.session_state.y, pred, prob)
    
    # Performance comparison
    leaderboard_df = leaderboard.get_leaderboard(sort_by="f1_score")
    
    # Key metrics comparison
    key_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'pr_auc']
    available_metrics = [m for m in key_metrics if m in leaderboard_df.columns]
    
    if available_metrics:
        fig = px.bar(
            leaderboard_df.melt(id_vars=['model_name'], value_vars=available_metrics),
            x='model_name', y='value', color='variable',
            title='Key Metrics Comparison',
            labels={'model_name': 'Model', 'value': 'Score', 'variable': 'Metric'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Security metrics
    st.subheader("Security Metrics")
    
    security_metrics = ['true_positive_rate', 'false_positive_rate', 'alert_rate']
    available_security = [m for m in security_metrics if m in leaderboard_df.columns]
    
    if available_security:
        fig = px.bar(
            leaderboard_df.melt(id_vars=['model_name'], value_vars=available_security),
            x='model_name', y='value', color='variable',
            title='Security Metrics Comparison',
            labels={'model_name': 'Model', 'value': 'Rate', 'variable': 'Metric'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Best model analysis
    best_model = leaderboard.get_best_model("f1_score")
    if best_model:
        st.subheader(f"Best Model: {best_model['model_name']}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("F1 Score", f"{best_model['metrics']['f1_score']:.3f}")
            st.metric("Precision", f"{best_model['metrics']['precision']:.3f}")
            st.metric("Recall", f"{best_model['metrics']['recall']:.3f}")
        
        with col2:
            st.metric("ROC AUC", f"{best_model['metrics']['roc_auc']:.3f}")
            st.metric("PR AUC", f"{best_model['metrics']['pr_auc']:.3f}")
            st.metric("Accuracy", f"{best_model['metrics']['accuracy']:.3f}")
    
    # Feature analysis
    st.subheader("Feature Analysis")
    
    # Get feature importance from best model
    if best_model and best_model['model_name'] in st.session_state.trained_models:
        model = st.session_state.trained_models[best_model['model_name']]
        
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': st.session_state.feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=True)
            
            fig = px.bar(
                importance_df, x='importance', y='feature',
                title='Feature Importance (Best Model)',
                labels={'importance': 'Importance', 'feature': 'Feature'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Recommendations
    st.subheader("Recommendations")
    
    recommendations = []
    
    # Model performance recommendations
    if best_model:
        f1_score = best_model['metrics']['f1_score']
        if f1_score < 0.7:
            recommendations.append("🔴 Model performance is below 70% F1 score. Consider feature engineering or model tuning.")
        elif f1_score < 0.8:
            recommendations.append("🟡 Model performance is moderate. Consider ensemble methods or hyperparameter tuning.")
        else:
            recommendations.append("🟢 Model performance is good. Consider deployment with monitoring.")
    
    # Security recommendations
    if 'defense_results' in st.session_state:
        results_df = st.session_state.defense_results
        block_rate = len(results_df[results_df['is_allowed'] == False]) / len(results_df)
        
        if block_rate < 0.1:
            recommendations.append("🟡 Low block rate. Consider tightening defense parameters.")
        elif block_rate > 0.3:
            recommendations.append("🔴 High block rate. Consider relaxing defense parameters to reduce false positives.")
        else:
            recommendations.append("🟢 Defense system is well-calibrated.")
    
    # General recommendations
    recommendations.extend([
        "📊 Monitor model performance over time and retrain periodically.",
        "🛡️ Implement multiple defense layers for robust protection.",
        "📈 Use ensemble methods to improve detection accuracy.",
        "🔍 Regularly analyze false positives and false negatives.",
        "⚡ Consider real-time monitoring and alerting systems."
    ])
    
    for rec in recommendations:
        st.write(rec)
    
    # Export results
    st.subheader("Export Results")
    
    if st.button("Export Leaderboard"):
        csv = leaderboard_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="model_stealing_detection_leaderboard.csv",
            mime="text/csv"
        )


if __name__ == "__main__":
    main()
