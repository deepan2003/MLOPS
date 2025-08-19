from flask import Flask, request, jsonify, render_template_string
import joblib
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables for model artifacts
model = None
model_metadata = None

# Get the directory paths
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(APP_DIR))

def load_model_artifacts():
    """Load model and metadata at startup."""
    global model, model_metadata
    
    try:
        # Use absolute paths based on project root
        model_path = os.path.join(PROJECT_ROOT, 'models', 'heart_attack_model.pkl')
        metrics_path = os.path.join(PROJECT_ROOT, 'models', 'metrics.json')
        
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            logger.info("✅ Model loaded successfully")
        else:
            logger.error(f"❌ Model file not found: {model_path}")
            return False
        
        # Load metadata
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                model_metadata = json.load(f)
            logger.info("✅ Model metadata loaded successfully")
        else:
            logger.warning("⚠️ Model metadata not found")
            model_metadata = {}
        
        return True
    except Exception as e:
        logger.error(f"❌ Error loading model artifacts: {e}")
        return False

# Load model at startup
model_loaded = load_model_artifacts()

# Required features for prediction
REQUIRED_FEATURES = ["age", "gender", "pressurehight", "pressurelow", "glucose", "kcm", "troponin"]

@app.route('/')
def home():
    """Home page with interactive prediction form."""
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Heart Attack Prediction API</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            * {
                box-sizing: border-box;
                margin: 0;
                padding: 0;
            }
            
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background-color: #f8f9fa;
                padding: 20px;
                line-height: 1.6;
            }
            
            .container {
                max-width: 600px;
                margin: 0 auto;
                background: white;
                padding: 30px;
                border-radius: 15px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            }
            
            .header {
                text-align: center;
                margin-bottom: 30px;
            }
            
            .header h1 {
                color: #2c3e50;
                font-size: 28px;
                margin-bottom: 10px;
            }
            
            .header p {
                color: #7f8c8d;
                font-size: 16px;
            }
            
            .status {
                padding: 15px;
                border-radius: 8px;
                margin-bottom: 25px;
                text-align: center;
                font-weight: 500;
            }
            
            .status.healthy {
                background-color: #d4edda;
                color: #155724;
                border: 1px solid #c3e6cb;
            }
            
            .status.unhealthy {
                background-color: #f8d7da;
                color: #721c24;
                border: 1px solid #f5c6cb;
            }
            
            .form-group {
                margin-bottom: 20px;
            }
            
            .form-row {
                display: flex;
                gap: 15px;
                margin-bottom: 20px;
            }
            
            .form-row .form-group {
                flex: 1;
                margin-bottom: 0;
            }
            
            label {
                display: block;
                font-weight: 600;
                color: #2c3e50;
                margin-bottom: 8px;
                font-size: 14px;
            }
            
            .range-info {
                font-size: 12px;
                color: #6c757d;
                margin-top: 2px;
            }
            
            input[type="number"] {
                width: 100%;
                padding: 12px;
                border: 2px solid #e9ecef;
                border-radius: 8px;
                font-size: 16px;
                transition: all 0.3s ease;
                background-color: #f8f9fa;
            }
            
            input[type="number"]:focus {
                outline: none;
                border-color: #007bff;
                background-color: white;
                box-shadow: 0 0 0 3px rgba(0,123,255,0.1);
            }
            
            input[type="number"]:invalid {
                border-color: #dc3545;
            }
            
            .predict-btn {
                width: 100%;
                padding: 15px;
                background: linear-gradient(135deg, #007bff, #0056b3);
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 18px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
                margin-top: 10px;
            }
            
            .predict-btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 25px rgba(0,123,255,0.3);
            }
            
            .predict-btn:active {
                transform: translateY(0);
            }
            
            .predict-btn:disabled {
                background: #6c757d;
                cursor: not-allowed;
                transform: none;
                box-shadow: none;
            }
            
            .result {
                margin-top: 25px;
                padding: 20px;
                border-radius: 8px;
                display: none;
                text-align: center;
            }
            
            .result.success {
                background-color: #d1ecf1;
                border: 1px solid #bee5eb;
                color: #0c5460;
            }
            
            .result.error {
                background-color: #f8d7da;
                border: 1px solid #f5c6cb;
                color: #721c24;
            }
            
            .result h3 {
                margin-bottom: 15px;
                font-size: 20px;
            }
            
            .result p {
                margin-bottom: 8px;
                font-size: 16px;
            }
            
            .loading {
                display: inline-block;
                width: 20px;
                height: 20px;
                border: 3px solid #f3f3f3;
                border-top: 3px solid #007bff;
                border-radius: 50%;
                animation: spin 1s linear infinite;
                margin-right: 10px;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            .risk-high {
                color: #dc3545;
                font-weight: bold;
            }
            
            .risk-low {
                color: #28a745;
                font-weight: bold;
            }
            
            @media (max-width: 600px) {
                .form-row {
                    flex-direction: column;
                    gap: 0;
                }
                
                .container {
                    padding: 20px;
                    margin: 10px;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🏥 Heart Attack Prediction API</h1>
                <p>Machine Learning API for Heart Attack Risk Assessment</p>
            </div>
            
            <div class="status {{ 'healthy' if model_loaded else 'unhealthy' }}">
                <strong>Status:</strong> {{ 'Model Loaded Successfully ✅' if model_loaded else 'Model Not Available ❌' }}
                {% if model_metadata and model_metadata.get('performance_metrics') %}
                <br><strong>Model Accuracy:</strong> {{ "%.3f"|format(model_metadata.performance_metrics.test_accuracy) }}
                <br><strong>Model F1-Score:</strong> {{ "%.3f"|format(model_metadata.performance_metrics.test_f1) }}
                {% endif %}
            </div>
            
            <form id="predictionForm">
                <div class="form-group">
                    <label for="age">Age</label>
                    <input type="number" id="age" name="age" min="14" max="103" required>
                    <div class="range-info">Valid range: 14-103 years</div>
                </div>
                
                <div class="form-group">
                    <label for="gender">Gender</label>
                    <input type="number" id="gender" name="gender" min="0" max="1" required>
                    <div class="range-info">0 = Female, 1 = Male</div>
                </div>
                
                <div class="form-group">
                    <label for="pressurehight">Systolic Blood Pressure (High)</label>
                    <input type="number" id="pressurehight" name="pressurehight" min="42" max="223" required>
                    <div class="range-info">Valid range: 42-223 mmHg</div>
                </div>
                
                <div class="form-group">
                    <label for="pressurelow">Diastolic Blood Pressure (Low)</label>
                    <input type="number" id="pressurelow" name="pressurelow" min="38" max="154" required>
                    <div class="range-info">Valid range: 38-154 mmHg</div>
                </div>
                
                <div class="form-group">
                    <label for="glucose">Blood Glucose Level</label>
                    <input type="number" id="glucose" name="glucose" min="35" max="541" step="0.1" required>
                    <div class="range-info">Valid range: 35-541 mg/dL</div>
                </div>
                
                <div class="form-row">
                    <div class="form-group">
                        <label for="kcm">CPK-MB Enzyme Level (KCM)</label>
                        <input type="number" id="kcm" name="kcm" min="0.321" max="300" step="0.001" required>
                        <div class="range-info">Range: 0.321-300 ng/mL</div>
                    </div>
                    
                    <div class="form-group">
                        <label for="troponin">Troponin Level</label>
                        <input type="number" id="troponin" name="troponin" min="0.001" max="10.3" step="0.001" required>
                        <div class="range-info">Range: 0.001-10.3 ng/mL</div>
                    </div>
                </div>
                
                <button type="submit" class="predict-btn" id="predictBtn">
                    Predict Risk
                </button>
            </form>
            
            <div id="result" class="result"></div>
        </div>
        
        <script>
            document.getElementById('predictionForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const form = e.target;
                const formData = new FormData(form);
                const resultDiv = document.getElementById('result');
                const predictBtn = document.getElementById('predictBtn');
                
                // Show loading state
                predictBtn.disabled = true;
                predictBtn.innerHTML = '<span class="loading"></span>Predicting...';
                resultDiv.style.display = 'none';
                
                // Prepare data
                const data = {};
                for (let [key, value] of formData.entries()) {
                    data[key] = parseFloat(value);
                }
                
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify(data)
                    });
                    
                    const result = await response.json();
                    
                    if (response.ok) {
                        const riskClass = result.heart_attack_risk === 'High Risk' ? 'risk-high' : 'risk-low';
                        const confidencePercent = (result.confidence * 100).toFixed(1);
                        
                        resultDiv.innerHTML = `
                            <h3>🔍 Prediction Result</h3>
                            <p><strong>Risk Level:</strong> <span class="${riskClass}">${result.heart_attack_risk}</span></p>
                            <p><strong>Confidence:</strong> ${confidencePercent}%</p>
                            <p><strong>Prediction:</strong> ${result.prediction}</p>
                        `;
                        resultDiv.className = 'result success';
                    } else {
                        resultDiv.innerHTML = `
                            <h3>❌ Error</h3>
                            <p>${result.error || 'An error occurred during prediction'}</p>
                        `;
                        resultDiv.className = 'result error';
                    }
                } catch (error) {
                    resultDiv.innerHTML = `
                        <h3>❌ Network Error</h3>
                        <p>Failed to connect to the prediction service. Please try again.</p>
                    `;
                    resultDiv.className = 'result error';
                } finally {
                    // Reset button
                    predictBtn.disabled = false;
                    predictBtn.innerHTML = 'Predict Risk';
                    resultDiv.style.display = 'block';
                }
            });
            
            // Add real-time validation feedback
            const inputs = document.querySelectorAll('input[type="number"]');
            inputs.forEach(input => {
                input.addEventListener('input', function() {
                    const min = parseFloat(this.min);
                    const max = parseFloat(this.max);
                    const value = parseFloat(this.value);
                    
                    if (value < min || value > max) {
                        this.style.borderColor = '#dc3545';
                    } else {
                        this.style.borderColor = '#28a745';
                    }
                });
            });
        </script>
    </body>
    </html>
    """
    
    return render_template_string(
        html_template,
        model_loaded=model_loaded,
        model_metadata=model_metadata
    )

@app.route('/predict', methods=['POST'])
def predict():
    """Single patient prediction endpoint - COMPLETELY FIXED."""
    try:
        if not model_loaded:
            return jsonify({'error': 'Model not available'}), 503
        
        # Get input data
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Validate required features
        missing_features = set(REQUIRED_FEATURES) - set(data.keys())
        if missing_features:
            return jsonify({
                'error': f'Missing required features: {list(missing_features)}',
                'required_features': REQUIRED_FEATURES,
                'provided_features': list(data.keys())
            }), 400
        
        # Prepare input for prediction
        input_df = pd.DataFrame([data])
        
        # Make predictions
        prediction_array = model.predict(input_df[REQUIRED_FEATURES])
        probabilities_array = model.predict_proba(input_df[REQUIRED_FEATURES])
        
        # ✅ FIXED: Extract single values correctly with [0] index
        prediction = prediction_array[0]       # Extract 'positive' from ['positive']
        probabilities = probabilities_array[0] # ← FIXED: Added  here!
        
        # ✅ FIXED: Now probabilities is 1D array, so this works:
        result = {
            'prediction': str(prediction),
            'heart_attack_risk': 'High Risk' if prediction == 'positive' else 'Low Risk',
            'probabilities': {
                'negative': float(probabilities[0]),    # Works: 0.3 → float
                'positive': float(probabilities[1])     # Works: 0.7 → float
            },
            'confidence': float(np.max(probabilities)), # Works: max([0.3, 0.7]) → float
            'input_data': data,
            'timestamp': datetime.now().isoformat()
        }
        
        # Log prediction
        logger.info(f"Prediction made: {prediction}, confidence: {result['confidence']:.3f}")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    """Health check endpoint."""
    health_status = {
        'status': 'healthy' if model_loaded else 'unhealthy',
        'model_loaded': model_loaded,
        'timestamp': datetime.now().isoformat(),
        'api_version': '1.0.0'
    }
    
    status_code = 200 if model_loaded else 503
    return jsonify(health_status), status_code

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') != 'production'
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug
    )
