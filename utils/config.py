import os
from pathlib import Path

class Config:
    # Paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / 'data'
    MODELS_DIR = BASE_DIR / 'models'
    
    # Data settings
    DATA_URL = "https://raw.githubusercontent.com/your-repo/supply-chain-data/main/supply_chain_data.csv"
    
    # Model settings
    TFT_CONFIG = {
        'max_encoder_length': 30,
        'max_prediction_length': 7,
        'hidden_size': 128,
        'attention_head_size': 4,
        'dropout': 0.1,
        'hidden_continuous_size': 64,
    }
    
    LSTM_CONFIG = {
        'sequence_length': 30,
        'n_features': 1,
        'lstm_units': [64, 32],
        'dropout': 0.2,
        'epochs': 50,
        'batch_size': 32,
    }
    
    # SHAP settings
    SHAP_CONFIG = {
        'n_samples': 500,
        'feature_names': [],
    }