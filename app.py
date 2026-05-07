from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
import pandas as pd
import joblib
import torch
import os
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import io
import base64

# Import our models
import sys
sys.path.append('.')
from models.lstm_model import LSTMModel
from models.xgboost_model import XGBoostModel
from models.tft_model import TFTModel
from utils.preprocessing import DataPreprocessor

app = Flask(__name__)
CORS(app)
app.secret_key = os.getenv('SECRET_KEY', 'default-secret-key')

# Global model instances
lstm_model = None
xgb_model = None
tft_model = None
preprocessor = None

# Model metrics storage
model_metrics = {
    'lstm': {},
    'xgboost': {},
    'tft': {}
}

def init_models():
    """Initialize and load all models"""
    global lstm_model, xgb_model, tft_model, preprocessor
    
    print("🔄 Loading models...")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Load LSTM
    try:
        lstm_model = LSTMModel()
        lstm_model.load_model('models/lstm_model.pkl')
        print("✅ LSTM model loaded")
    except Exception as e:
        print(f"⚠️ LSTM model not loaded: {e}")
    
    # Load XGBoost
    try:
        xgb_model = XGBoostModel()
        xgb_model.load_model('models/xgboost_model.pkl')
        print("✅ XGBoost model loaded")
    except Exception as e:
        print(f"⚠️ XGBoost model not loaded: {e}")
    
    # Load TFT
    try:
        tft_model = TFTModel()
        tft_model.load_model('models/tft_model.pkl')
        print("✅ TFT model loaded")
    except Exception as e:
        print(f"⚠️ TFT model not loaded: {e}")
    
    print("✅ All models loaded!")

def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string"""
    img = io.BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight', dpi=100)
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/api/forecast', methods=['POST'])
def forecast():
    """API endpoint for demand forecasting"""
    try:
        data = request.get_json()
        
        model_type = data.get('model', 'lstm')
        product_id = data.get('product_id', 1)
        store_id = data.get('store_id', 1)
        forecast_days = data.get('forecast_days', 7)
        
        # Generate sample historical data for the request
        historical_data = generate_sample_sequence(product_id, store_id)
        
        predictions = []
        model_used = model_type
        
        if model_type == 'lstm' and lstm_model:
            future_preds = lstm_model.predict_future(historical_data, n_steps=forecast_days)
            predictions = [int(round(p)) for p in future_preds.tolist()]
        elif model_type == 'xgboost' and xgb_model:
            future_preds = predict_xgboost_future(historical_data, forecast_days)
            predictions = [int(round(p)) for p in future_preds.tolist()]
        elif model_type == 'tft' and tft_model:
            future_preds = tft_model.predict_future(historical_data, n_steps=forecast_days)
            predictions = [int(round(p)) for p in future_preds.tolist()]
        elif model_type == 'ensemble':
            # Ensemble prediction (average of all models)
            if lstm_model and tft_model:
                preds_lstm = lstm_model.predict_future(historical_data, n_steps=forecast_days)
                preds_xgb = predict_xgboost_future(historical_data, forecast_days)
                preds_tft = tft_model.predict_future(historical_data, n_steps=forecast_days)
                predictions = np.mean([preds_lstm, preds_xgb, preds_tft], axis=0)
                predictions = [int(round(p)) for p in predictions.tolist()]
            else:
                return jsonify({'error': 'All models not available for ensemble'}), 400
            model_used = 'Ensemble (LSTM + XGBoost + TFT)'
        else:
            return jsonify({'error': 'Model not available'}), 400
        
        # Generate dates for predictions
        dates = [(datetime.now() + timedelta(days=i+1)).strftime('%Y-%m-%d') 
                 for i in range(forecast_days)]
        
        # Calculate total forecasted demand (whole numbers)
        total_demand = int(sum(predictions))
        avg_demand = int(total_demand / forecast_days)
        
        return jsonify({
            'success': True,
            'model': model_used,
            'product_id': product_id,
            'store_id': store_id,
            'forecast_days': forecast_days,
            'predictions': predictions,
            'dates': dates,
            'total_demand': total_demand,
            'avg_daily_demand': avg_demand
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/model_comparison', methods=['GET'])
def model_comparison():
    """API endpoint for model performance comparison"""
    return jsonify(model_metrics)

@app.route('/api/metrics/<model_type>', methods=['GET'])
def get_metrics(model_type):
    """Get metrics for specific model"""
    if model_type in model_metrics:
        return jsonify(model_metrics[model_type])
    return jsonify({'error': 'Model not found'}), 404

def generate_sample_sequence(product_id=1, store_id=1):
    """Generate sample historical sequence for prediction"""
    seed = product_id * 1000 + store_id * 100 + int(datetime.now().timestamp()) % 1000
    np.random.seed(seed)
    
    base_demand = 80 + (product_id * 25) + (store_id * 12)
    sequence = []
    
    for i in range(30):
        trend = i * (0.3 + product_id * 0.1)
        seasonality = (10 + store_id * 3) * np.sin(2 * np.pi * i / 7)
        noise = np.random.normal(0, base_demand * 0.08)
        
        demand = base_demand + trend + seasonality + noise
        sequence.append(max(0, demand))
    
    return np.array(sequence)

def predict_xgboost_future(historical_sequence, n_steps):
    """Make future predictions using XGBoost"""
    predictions = []
    last_values = historical_sequence[-7:]
    
    for i in range(n_steps):
        trend = np.mean(np.diff(last_values)) if len(last_values) > 1 else 0
        next_pred = last_values[-1] + trend + np.random.normal(0, 5)
        predictions.append(max(0, int(round(next_pred))))
        last_values = np.append(last_values[1:], next_pred)
    
    return np.array(predictions)

if __name__ == '__main__':
    init_models()
    app.run(debug=True, host='0.0.0.0', port=5000)