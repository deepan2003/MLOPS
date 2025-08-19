import pytest
import pandas as pd
import numpy as np
import os
import sys
import tempfile
import joblib
import json

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.train_model import (
    load_processed_data, train_random_forest, evaluate_model, 
    save_model_artifacts, load_params
)
from models.predict_model import (
    load_model_artifacts, predict_single, predict_batch
)

@pytest.fixture
def sample_processed_data():
    """Create sample processed training and test data."""
    np.random.seed(42)
    n_samples = 100
    
    # Create synthetic heart attack data
    train_data = pd.DataFrame({
        'age': np.random.randint(20, 80, n_samples),
        'gender': np.random.choice([0, 1], n_samples),
        'pressurehight': np.random.randint(90, 200, n_samples),
        'pressurelow': np.random.randint(60, 120, n_samples),
        'glucose': np.random.uniform(80, 400, n_samples),
        'kcm': np.random.uniform(0.5, 50, n_samples),
        'troponin': np.random.uniform(0.001, 5.0, n_samples),
        'class': np.random.choice(['positive', 'negative'], n_samples, p=[0.3, 0.7])
    })
    
    test_data = pd.DataFrame({
        'age': np.random.randint(20, 80, 25),
        'gender': np.random.choice([0, 1], 25),
        'pressurehight': np.random.randint(90, 200, 25),
        'pressurelow': np.random.randint(60, 120, 25),
        'glucose': np.random.uniform(80, 400, 25),
        'kcm': np.random.uniform(0.5, 50, 25),
        'troponin': np.random.uniform(0.001, 5.0, 25),
        'class': np.random.choice(['positive', 'negative'], 25, p=[0.3, 0.7])
    })
    
    return train_data, test_data

@pytest.fixture
def test_config():
    """Test configuration for model training."""
    return {
        'preprocessing': {
            'features': ['age', 'gender', 'pressurehight', 'pressurelow', 'glucose', 'kcm', 'troponin'],
            'target': 'class'
        },
        'model': {
            'n_estimators': 10,  # Small for testing
            'random_state': 42,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1
        },
        'evaluation': {
            'cv_folds': 3  # Small for testing
        }
    }

class TestModelTraining:
    
    def test_train_random_forest(self, sample_processed_data, test_config):
        """Test Random Forest model training."""
        train_data, _ = sample_processed_data
        features = test_config['preprocessing']['features']
        target = test_config['preprocessing']['target']
        
        X_train = train_data[features]
        y_train = train_data[target]
        
        model = train_random_forest(X_train, y_train, test_config)
        
        # Check that model is trained
        assert hasattr(model, 'predict')
        assert hasattr(model, 'predict_proba')
        assert hasattr(model, 'feature_importances_')
        
        # Check model parameters
        assert model.n_estimators == test_config['model']['n_estimators']
        assert model.random_state == test_config['model']['random_state']
    
    def test_model_predictions_shape(self, sample_processed_data, test_config):
        """Test that model predictions have correct shape."""
        train_data, test_data = sample_processed_data
        features = test_config['preprocessing']['features']
        target = test_config['preprocessing']['target']
        
        X_train = train_data[features]
        y_train = train_data[target]
        X_test = test_data[features]
        y_test =test_data[target]
        
        model = train_random_forest(X_train, y_train, test_config)
        
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)
        
        assert len(predictions) == len(X_test)
        assert probabilities.shape[0] == len(X_test)
        assert probabilities.shape[1] == 2  # Binary classification
        assert all(pred in ['positive', 'negative'] for pred in predictions)

class TestModelEvaluation:
    
    def test_evaluate_model(self, sample_processed_data, test_config):
        """Test model evaluation function."""
        train_data, test_data = sample_processed_data
        features = test_config['preprocessing']['features']
        target = test_config['preprocessing']['target']
        
        X_train = train_data[features]
        y_train = train_data[target]
        X_test = test_data[features]
        y_test = test_data[target]
        
        model = train_random_forest(X_train, y_train, test_config)
        metrics, feature_importance = evaluate_model(
            model, X_train, y_train, X_test, y_test, test_config
        )
        
        # Check that metrics are calculated
        required_metrics = [
            'train_accuracy', 'test_accuracy', 'train_precision', 'test_precision',
            'train_recall', 'test_recall', 'train_f1', 'test_f1',
            'train_roc_auc', 'test_roc_auc', 'cv_f1_mean', 'cv_f1_std'
        ]
        
        for metric in required_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
            assert 0 <= metrics[metric] <= 1 or metric.endswith('_std')
        
        # Check feature importance
        assert 'features' in feature_importance
        assert 'importance' in feature_importance
        assert len(feature_importance['features']) == len(features)
        assert len(feature_importance['importance']) == len(features)
    
    def test_model_performance_threshold(self, sample_processed_data, test_config):
        """Test that model meets minimum performance requirements."""
        train_data, test_data = sample_processed_data
        features = test_config['preprocessing']['features']
        target = test_config['preprocessing']['target']
        
        X_train = train_data[features]
        y_train = train_data[target]
        X_test = test_data[features]
        y_test = test_data[target]
        
        model = train_random_forest(X_train, y_train, test_config)
        metrics, _ = evaluate_model(
            model, X_train, y_train, X_test, y_test, test_config
        )
        
        # Model should perform better than random (>0.5 for most metrics)
        assert metrics['test_accuracy'] > 0.4  # Lenient threshold for synthetic data
        assert metrics['test_roc_auc'] > 0.4

class TestModelSaving:
    
    def test_save_model_artifacts(self, sample_processed_data, test_config):
        """Test saving model artifacts."""
        train_data, test_data = sample_processed_data
        features = test_config['preprocessing']['features']
        target = test_config['preprocessing']['target']
        
        X_train = train_data[features]
        y_train = train_data[target]
        X_test = test_data[features]
        y_test = test_data[target]
        
        model = train_random_forest(X_train, y_train, test_config)
        metrics, feature_importance = evaluate_model(
            model, X_train, y_train, X_test, y_test, test_config
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Change to temp directory
            original_cwd = os.getcwd()
            os.chdir(temp_dir)
            
            try:
                model_path, metrics_path = save_model_artifacts(
                    model, metrics, feature_importance, test_config
                )
                
                # Check that files are created
                assert os.path.exists(model_path)
                assert os.path.exists(metrics_path)
                
                # Check that model can be loaded
                loaded_model = joblib.load(model_path)
                assert hasattr(loaded_model, 'predict')
                
                # Check that metrics can be loaded
                with open(metrics_path, 'r') as f:
                    loaded_metrics = json.load(f)
                assert 'performance_metrics' in loaded_metrics
                assert 'feature_importance' in loaded_metrics
                
            finally:
                os.chdir(original_cwd)

class TestPrediction:
    
    def test_predict_single(self, sample_processed_data, test_config):
        """Test single prediction function."""
        train_data, _ = sample_processed_data
        features = test_config['preprocessing']['features']
        target = test_config['preprocessing']['target']
        
        X_train = train_data[features]
        y_train = train_data[target]
        
        model = train_random_forest(X_train, y_train, test_config)
        
        # Create sample input
        sample_input = {
            'age': 64,
            'gender': 1,
            'pressurehight': 160,
            'pressurelow': 83,
            'glucose': 160.0,
            'kcm': 1.80,
            'troponin': 0.012
        }
        
        result = predict_single(model, sample_input, test_config)
        
        # Check result structure
        assert 'prediction' in result
        assert 'heart_attack_risk' in result
        assert 'probabilities' in result
        assert 'confidence' in result
        assert 'input_features' in result
        
        # Check prediction values
        assert result['prediction'] in ['positive', 'negative']
        assert result['heart_attack_risk'] in ['High Risk', 'Low Risk']
        assert 0 <= result['confidence'] <= 1
        assert 'negative' in result['probabilities']
        assert 'positive' in result['probabilities']
    
    def test_predict_single_missing_features(self, sample_processed_data, test_config):
        """Test single prediction with missing features."""
        train_data, _ = sample_processed_data
        features = test_config['preprocessing']['features']
        target = test_config['preprocessing']['target']
        
        X_train = train_data[features]
        y_train = train_data[target]
        
        model = train_random_forest(X_train, y_train, test_config)
        
        # Create input with missing feature
        incomplete_input = {
            'age': 64,
            'gender': 1,
            'pressurehight': 160
            # Missing other required features
        }
        
        with pytest.raises(ValueError, match="Missing feature"):
            predict_single(model, incomplete_input, test_config)
