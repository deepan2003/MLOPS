import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, classification_report, 
                           confusion_matrix)
from sklearn.model_selection import cross_val_score
import joblib
import yaml
import json
import os
from datetime import datetime

def load_params(config_path="params.yaml"):
    """Load parameters from YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)

def load_processed_data(train_path, test_path, config):
    """Load processed training and test data."""
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    features = config['preprocessing']['features']
    target = config['preprocessing']['target']
    
    X_train = train_data[features]
    y_train = train_data[target]
    X_test = test_data[features]
    y_test = test_data[target]
    
    print(f"✅ Loaded training data: {X_train.shape}")
    print(f"✅ Loaded test data: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

def train_random_forest(X_train, y_train, config):
    """Train Random Forest model with specified parameters."""
    print("\n🤖 Training Random Forest model...")
    
    # Get model parameters from config
    model_params = {
        'n_estimators': config['model']['n_estimators'],
        'random_state': config['model']['random_state']
    }
    
    # Add optional parameters if they exist and are not null
    if config['model']['max_depth'] is not None:
        model_params['max_depth'] = config['model']['max_depth']
    if config['model']['min_samples_split'] != 2:
        model_params['min_samples_split'] = config['model']['min_samples_split']
    if config['model']['min_samples_leaf'] != 1:
        model_params['min_samples_leaf'] = config['model']['min_samples_leaf']
    
    print(f"Model parameters: {model_params}")
    
    # Create and train model
    model = RandomForestClassifier(**model_params)
    model.fit(X_train, y_train)
    
    print("✅ Model training completed!")
    return model

def evaluate_model(model, X_train, y_train, X_test, y_test, config):
    """Evaluate model performance."""
    print("\n📊 Evaluating model performance...")
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Get prediction probabilities for ROC-AUC
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'train_accuracy': accuracy_score(y_train, y_train_pred),
        'test_accuracy': accuracy_score(y_test, y_test_pred),
        'train_precision': precision_score(y_train, y_train_pred, pos_label='positive'),
        'test_precision': precision_score(y_test, y_test_pred, pos_label='positive'),
        'train_recall': recall_score(y_train, y_train_pred, pos_label='positive'),
        'test_recall': recall_score(y_test, y_test_pred, pos_label='positive'),
        'train_f1': f1_score(y_train, y_train_pred, pos_label='positive'),
        'test_f1': f1_score(y_test, y_test_pred, pos_label='positive'),
        'train_roc_auc': roc_auc_score(y_train == 'positive', y_train_proba),
        'test_roc_auc': roc_auc_score(y_test == 'positive', y_test_proba)
    }
    
    # Cross-validation scores
    cv_scores = cross_val_score(
        model, X_train, y_train, 
        cv=config['evaluation']['cv_folds'], 
        scoring='f1_macro'
    )
    metrics['cv_f1_mean'] = cv_scores.mean()
    metrics['cv_f1_std'] = cv_scores.std()
    
    # Print results
    print(f"Training Accuracy: {metrics['train_accuracy']:.4f}")
    print(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"Test Precision: {metrics['test_precision']:.4f}")
    print(f"Test Recall: {metrics['test_recall']:.4f}")
    print(f"Test F1-Score: {metrics['test_f1']:.4f}")
    print(f"Test ROC-AUC: {metrics['test_roc_auc']:.4f}")
    print(f"CV F1-Score: {metrics['cv_f1_mean']:.4f} (+/- {metrics['cv_f1_std']*2:.4f})")
    
    # Classification report
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_test_pred))
    
    # Confusion Matrix
    print("\n🔍 Confusion Matrix:")
    cm = confusion_matrix(y_test, y_test_pred)
    print(cm)
    
    # Feature importance
    feature_importance = {
        'features': config['preprocessing']['features'],
        'importance': model.feature_importances_.tolist()
    }
    
    return metrics, feature_importance

def save_model_artifacts(model, metrics, feature_importance, config):
    """Save model and related artifacts."""
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save model
    model_path = 'models/heart_attack_model.pkl'
    joblib.dump(model, model_path)
    print(f"✅ Model saved: {model_path}")
    
    # Save metrics
    metrics_data = {
        'model_type': 'RandomForestClassifier',
        'training_date': datetime.now().isoformat(),
        'model_parameters': {
            'n_estimators': config['model']['n_estimators'],
            'random_state': config['model']['random_state']
        },
        'performance_metrics': metrics,
        'feature_importance': feature_importance
    }
    
    metrics_path = 'models/metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    print(f"✅ Metrics saved: {metrics_path}")
    
    return model_path, metrics_path

def main():
    """Main training pipeline."""
    print("🚀 Starting Heart Attack Model Training Pipeline")
    
    # Load configuration
    config = load_params()
    
    # Load processed data
    train_path = config['data']['train_path']
    test_path = config['data']['test_path']
    
    X_train, X_test, y_train, y_test = load_processed_data(train_path, test_path, config)
    
    # Train model
    model = train_random_forest(X_train, y_train, config)
    
    # Evaluate model
    metrics, feature_importance = evaluate_model(
        model, X_train, y_train, X_test, y_test, config
    )
    
    # Save model artifacts
    model_path, metrics_path = save_model_artifacts(
        model, metrics, feature_importance, config
    )
    
    print("✅ Model training pipeline completed successfully!")
    return model_path, metrics_path

if __name__ == "__main__":
    main()
