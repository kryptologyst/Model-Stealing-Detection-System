"""Tests for detection models."""

import numpy as np
import pytest
from sklearn.datasets import make_classification

# Add src to path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.detection_models import (
    EntropyBasedDetector,
    IsolationForestDetector,
    OneClassSVMDetector,
    RandomForestDetector,
    NeuralNetworkDetector,
    EnsembleDetector
)


class TestEntropyBasedDetector:
    """Test entropy-based detector."""
    
    def test_init(self):
        """Test detector initialization."""
        detector = EntropyBasedDetector(threshold=2.0, n_bins=10)
        assert detector.name == "EntropyBasedDetector"
        assert detector.threshold == 2.0
        assert detector.n_bins == 10
        assert not detector.is_fitted
    
    def test_fit_predict(self):
        """Test fit and predict methods."""
        detector = EntropyBasedDetector(threshold=2.0)
        X = np.random.normal(0, 1, (100, 10))
        
        detector.fit(X)
        assert detector.is_fitted
        
        predictions = detector.predict(X)
        assert len(predictions) == len(X)
        assert all(pred in [0, 1] for pred in predictions)
    
    def test_predict_proba(self):
        """Test probability prediction."""
        detector = EntropyBasedDetector(threshold=2.0)
        X = np.random.normal(0, 1, (100, 10))
        
        detector.fit(X)
        probabilities = detector.predict_proba(X)
        
        assert probabilities.shape == (len(X), 2)
        assert np.allclose(probabilities.sum(axis=1), 1.0)


class TestIsolationForestDetector:
    """Test isolation forest detector."""
    
    def test_init(self):
        """Test detector initialization."""
        detector = IsolationForestDetector(contamination=0.1)
        assert detector.name == "IsolationForestDetector"
        assert detector.contamination == 0.1
        assert not detector.is_fitted
    
    def test_fit_predict(self):
        """Test fit and predict methods."""
        detector = IsolationForestDetector(contamination=0.1)
        X = np.random.normal(0, 1, (100, 10))
        
        detector.fit(X)
        assert detector.is_fitted
        
        predictions = detector.predict(X)
        assert len(predictions) == len(X)
        assert all(pred in [0, 1] for pred in predictions)
    
    def test_predict_proba(self):
        """Test probability prediction."""
        detector = IsolationForestDetector(contamination=0.1)
        X = np.random.normal(0, 1, (100, 10))
        
        detector.fit(X)
        probabilities = detector.predict_proba(X)
        
        assert probabilities.shape == (len(X), 2)
        assert np.allclose(probabilities.sum(axis=1), 1.0)


class TestOneClassSVMDetector:
    """Test one-class SVM detector."""
    
    def test_init(self):
        """Test detector initialization."""
        detector = OneClassSVMDetector(nu=0.1)
        assert detector.name == "OneClassSVMDetector"
        assert not detector.is_fitted
    
    def test_fit_predict(self):
        """Test fit and predict methods."""
        detector = OneClassSVMDetector(nu=0.1)
        X = np.random.normal(0, 1, (100, 10))
        
        detector.fit(X)
        assert detector.is_fitted
        
        predictions = detector.predict(X)
        assert len(predictions) == len(X)
        assert all(pred in [0, 1] for pred in predictions)
    
    def test_predict_proba(self):
        """Test probability prediction."""
        detector = OneClassSVMDetector(nu=0.1)
        X = np.random.normal(0, 1, (100, 10))
        
        detector.fit(X)
        probabilities = detector.predict_proba(X)
        
        assert probabilities.shape == (len(X), 2)
        assert np.allclose(probabilities.sum(axis=1), 1.0)


class TestRandomForestDetector:
    """Test random forest detector."""
    
    def test_init(self):
        """Test detector initialization."""
        detector = RandomForestDetector(n_estimators=10)
        assert detector.name == "RandomForestDetector"
        assert not detector.is_fitted
    
    def test_fit_predict(self):
        """Test fit and predict methods."""
        detector = RandomForestDetector(n_estimators=10)
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        
        detector.fit(X, y)
        assert detector.is_fitted
        
        predictions = detector.predict(X)
        assert len(predictions) == len(X)
        assert all(pred in [0, 1] for pred in predictions)
    
    def test_predict_proba(self):
        """Test probability prediction."""
        detector = RandomForestDetector(n_estimators=10)
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        
        detector.fit(X, y)
        probabilities = detector.predict_proba(X)
        
        assert probabilities.shape == (len(X), 2)
        assert np.allclose(probabilities.sum(axis=1), 1.0)


class TestNeuralNetworkDetector:
    """Test neural network detector."""
    
    def test_init(self):
        """Test detector initialization."""
        detector = NeuralNetworkDetector(input_dim=10, hidden_dims=[32, 16])
        assert detector.name == "NeuralNetworkDetector"
        assert detector.input_dim == 10
        assert detector.hidden_dims == [32, 16]
        assert not detector.is_fitted
    
    def test_fit_predict(self):
        """Test fit and predict methods."""
        detector = NeuralNetworkDetector(input_dim=10, hidden_dims=[32, 16])
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        
        detector.fit(X, y, epochs=5)  # Reduced epochs for testing
        assert detector.is_fitted
        
        predictions = detector.predict(X)
        assert len(predictions) == len(X)
        assert all(pred in [0, 1] for pred in predictions)
    
    def test_predict_proba(self):
        """Test probability prediction."""
        detector = NeuralNetworkDetector(input_dim=10, hidden_dims=[32, 16])
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        
        detector.fit(X, y, epochs=5)  # Reduced epochs for testing
        probabilities = detector.predict_proba(X)
        
        assert probabilities.shape == (len(X), 2)
        assert np.allclose(probabilities.sum(axis=1), 1.0)


class TestEnsembleDetector:
    """Test ensemble detector."""
    
    def test_init(self):
        """Test detector initialization."""
        model1 = RandomForestDetector(n_estimators=5)
        model2 = RandomForestDetector(n_estimators=5)
        detector = EnsembleDetector([model1, model2], [0.5, 0.5])
        
        assert detector.name == "EnsembleDetector"
        assert len(detector.models) == 2
        assert detector.weights == [0.5, 0.5]
        assert not detector.is_fitted
    
    def test_fit_predict(self):
        """Test fit and predict methods."""
        model1 = RandomForestDetector(n_estimators=5)
        model2 = RandomForestDetector(n_estimators=5)
        detector = EnsembleDetector([model1, model2], [0.5, 0.5])
        
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        
        detector.fit(X, y)
        assert detector.is_fitted
        
        predictions = detector.predict(X)
        assert len(predictions) == len(X)
        assert all(pred in [0, 1] for pred in predictions)
    
    def test_predict_proba(self):
        """Test probability prediction."""
        model1 = RandomForestDetector(n_estimators=5)
        model2 = RandomForestDetector(n_estimators=5)
        detector = EnsembleDetector([model1, model2], [0.5, 0.5])
        
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        
        detector.fit(X, y)
        probabilities = detector.predict_proba(X)
        
        assert probabilities.shape == (len(X), 2)
        assert np.allclose(probabilities.sum(axis=1), 1.0)


if __name__ == "__main__":
    pytest.main([__file__])
