# Model Stealing Detection System

A comprehensive research and educational framework for detecting model stealing attempts in machine learning systems. This project demonstrates various detection techniques, defense mechanisms, and evaluation metrics for protecting ML models from unauthorized replication.

## Overview

Model stealing detection aims to identify when users or attackers are querying a model in ways that suggest they're trying to replicate or reverse-engineer it. This system provides:

- **Multiple Detection Methods**: Entropy-based, isolation forest, one-class SVM, random forest, and neural network detectors
- **Advanced Feature Engineering**: Query pattern analysis, temporal features, and anomaly detection
- **Defense Mechanisms**: Rate limiting, query deduplication, response randomization, and honey traps
- **Comprehensive Evaluation**: Security-specific metrics, precision@K, recall@precision, and detection delay analysis
- **Explainability**: SHAP-based explanations and feature importance analysis
- **Interactive Demo**: Streamlit-based web interface for experimentation

## Disclaimer

**This is a research and educational demonstration only.** This system is designed for defensive purposes and should not be used for offensive activities or exploitation. See [DISCLAIMER.md](DISCLAIMER.md) for important limitations and legal considerations.

## Project Structure

```
├── src/                    # Source code
│   ├── data/              # Data generation and processing
│   ├── features/          # Feature engineering
│   ├── models/            # Detection models
│   ├── defenses/          # Defense mechanisms
│   ├── eval/              # Evaluation metrics
│   ├── viz/               # Visualization and explainability
│   └── utils/             # Utility functions
├── demo/                  # Interactive demo
├── configs/               # Configuration files
├── data/                  # Data storage
├── assets/                # Generated plots and artifacts
├── tests/                 # Unit tests
└── scripts/               # Utility scripts
```

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Model-Stealing-Detection-System.git
cd Model-Stealing-Detection-System
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the interactive demo:
```bash
streamlit run demo/app.py
```

### Basic Usage

```python
from src.data.synthetic import ModelStealingDataGenerator
from src.models.detection_models import EntropyBasedDetector
from src.eval.metrics import ModelStealingMetrics

# Generate synthetic data
generator = ModelStealingDataGenerator(seed=42)
df = generator.generate_mixed_dataset(n_legitimate=1000, n_stealing=200)

# Train detection model
detector = EntropyBasedDetector(threshold=2.0)
X = df[[col for col in df.columns if col.startswith('feature_')]].values
detector.fit(X)

# Make predictions
predictions = detector.predict(X)
probabilities = detector.predict_proba(X)

# Evaluate performance
metrics_calculator = ModelStealingMetrics()
metrics = metrics_calculator.calculate_all_metrics(df['is_stealing'], predictions, probabilities)
print(f"F1 Score: {metrics['f1_score']:.3f}")
```

## Features

### Detection Methods

1. **Entropy-Based Detector**: Uses query entropy to identify suspicious patterns
2. **Isolation Forest**: Anomaly detection for identifying unusual query behaviors
3. **One-Class SVM**: Unsupervised learning for detecting outliers
4. **Random Forest**: Supervised classification with feature importance
5. **Neural Network**: Deep learning approach for complex pattern recognition
6. **Ensemble Methods**: Combines multiple detectors for improved performance

### Feature Engineering

- **Basic Statistical Features**: Mean, std, min, max, percentiles
- **Distribution Features**: Skewness, kurtosis, normality tests
- **Temporal Features**: Query frequency, similarity, rolling statistics
- **Anomaly Features**: Isolation scores, LOF, OCSVM scores
- **Behavioral Features**: User patterns, session analysis

### Defense Mechanisms

- **Rate Limiting**: Prevents excessive API calls
- **Query Deduplication**: Detects repeated queries
- **Response Randomization**: Adds noise to model outputs
- **Honey Traps**: Special queries to detect attackers
- **API Defense System**: Comprehensive protection framework

### Evaluation Metrics

- **Traditional ML Metrics**: Accuracy, precision, recall, F1-score
- **Security Metrics**: TPR, FPR, alert rates, workload metrics
- **Precision@K**: Performance at different recall levels
- **Detection Delay**: Time to detect attacks
- **ROC/PR Curves**: Comprehensive performance visualization

## Configuration

The system uses YAML configuration files for easy customization:

```yaml
# configs/config.yaml
data:
  n_legitimate: 1000
  n_stealing: 200
  n_features: 10

models:
  entropy_detector:
    threshold: 2.0
  
  isolation_forest:
    contamination: 0.1

defenses:
  rate_limiter:
    max_requests_per_minute: 60
```

## Interactive Demo

The Streamlit demo provides an interactive interface for:

- **Data Generation**: Customize synthetic dataset parameters
- **Model Training**: Train and compare different detection methods
- **Results Analysis**: Visualize performance metrics and confusion matrices
- **Defense Simulation**: Test API defense mechanisms
- **Insights**: Get recommendations and export results

### Running the Demo

```bash
streamlit run demo/app.py
```

Navigate to `http://localhost:8501` to access the interactive interface.

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

## Documentation

- [Configuration Guide](configs/README.md)
- [API Reference](docs/api.md)
- [Evaluation Metrics](docs/metrics.md)
- [Defense Mechanisms](docs/defenses.md)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Research community for model stealing detection techniques
- Open source ML libraries (scikit-learn, PyTorch, etc.)
- Security research community for defense mechanisms

## Contact

- GitHub: [kryptologyst](https://github.com/kryptologyst)
- Project Issues: [GitHub Issues](https://github.com/kryptologyst/Model-Stealing-Detection-System/issues)

## Related Work

- [Model Extraction Attacks](https://arxiv.org/abs/1909.01838)
- [API Security for ML Models](https://arxiv.org/abs/2002.07669)
- [Defense Against Model Stealing](https://arxiv.org/abs/2104.15075)

---

**Remember**: This is a research and educational tool. Always use responsibly and in accordance with applicable laws and ethical guidelines.
# Model-Stealing-Detection-System
