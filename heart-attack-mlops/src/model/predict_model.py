import pandas as pd
import numpy as np
import joblib
import yaml
import json
from datetime import datetime

def load_params(config_path="params.yaml"):
    """Load parameters from YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)

def load_model_artifacts():
    """Load trained model and related artifacts."""
    try:
        # Load model
        model = joblib.load('models/heart_attack_model.pkl')
        
        # Load metrics
        with open('models/metrics.json', 'r') as f:
            metrics = json.load(f)
        
        print("✅ Model artifacts loaded successfully")
        return model, metrics
    except Exception as e:
        print(f"❌ Error loading model artifacts: {e}")
        raise

def predict_single(model, input_data, config):
    """Make prediction for single sample."""
    features = config['preprocessing']['features']
    
    # Ensure input data has all required features
    for feature in features:
        if feature not in input_data:
            raise ValueError(f"Missing feature: {feature}")
    
    # Create DataFrame with single row
    input_df = pd.DataFrame([input_data])
    
    # Make prediction
    prediction = model.predict(input_df[features])[0]
    probabilities = model.predict_proba(input_df[features])[0]
    
    # Create result
    result = {
        'prediction': prediction,
        'heart_attack_risk': 'High Risk' if prediction == 'positive' else 'Low Risk',
        'probabilities': {
            'negative': float(probabilities[0]),
            'positive': float(probabilities[1])
        },
        'confidence': float(max(probabilities)),
        'input_features': input_data,
        'prediction_timestamp': datetime.now().isoformat()
    }
    
    return result

def predict_batch(model, input_data, config):
    """Make predictions for batch of samples."""
    features = config['preprocessing']['features']
    
    # Ensure input data has all required features
    missing_features = set(features) - set(input_data.columns)
    if missing_features:
        raise ValueError(f"Missing features: {list(missing_features)}")
    
    # Make predictions
    predictions = model.predict(input_data[features])
    probabilities = model.predict_proba(input_data[features])
    
    # Create results
    results = []
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        result = {
            'sample_id': i,
            'prediction': pred,
            'heart_attack_risk': 'High Risk' if pred == 'positive' else 'Low Risk',
            'probabilities': {
                'negative': float(prob[0]),
                'positive': float(prob[1])
            },
            'confidence': float(max(prob))
        }
        results.append(result)
    
    batch_result = {
        'total_samples': len(results),
        'high_risk_count': sum(1 for r in results if r['prediction'] == 'positive'),
        'low_risk_count': sum(1 for r in results if r['prediction'] == 'negative'),
        'average_positive_probability': float(np.mean([r['probabilities']['positive'] for r in results])),
        'predictions': results,
        'prediction_timestamp': datetime.now().isoformat()
    }
    
    return batch_result

def main():
    """Example usage of prediction functions."""
    print("🔮 Heart Attack Prediction Example")
    
    # Load configuration and model
    config = load_params()
    model, metrics = load_model_artifacts()
    
    print(f"\nModel Performance:")
    print(f"Test Accuracy: {metrics['performance_metrics']['test_accuracy']:.4f}")
    print(f"Test F1-Score: {metrics['performance_metrics']['test_f1']:.4f}")
    
    # Example single prediction
    sample_patient = {
        "age": 64,
        "gender": 1,
        "pressurehight": 160,
        "pressurelow": 83,
        "glucose": 160.0,
        "kcm": 1.80,
        "troponin": 0.012
    }
    
    result = predict_single(model, sample_patient, config)
    print(f"\n🏥 Single Patient Prediction:")
    print(f"Risk Level: {result['heart_attack_risk']}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Positive Probability: {result['probabilities']['positive']:.3f}")
    
    return result

if __name__ == "__main__":
    main()
