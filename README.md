# 📦 Supply Chain Demand Forecasting

AI-Powered Demand Forecasting System using **LSTM**, **XGBoost with SHAP**, and **Temporal Fusion Transformer (TFT)**.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-red.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 🎯 Features

| Feature | Description |
|---------|-------------|
| **🔮 Multi-Model Forecasting** | LSTM, XGBoost, and TFT models for demand prediction |
| **🧠 SHAP Explainability** | Understand which features drive predictions |
| **👁️ Temporal Attention** | Visualize which time steps matter most |
| **🌐 Web Interface** | Interactive Flask dashboard for forecasting |
| **📊 Model Comparison** | Compare performance across all three models |
| **📈 1-30 Day Horizon** | Flexible forecast window |

---

## 🤖 AI Models

### 1. LSTM (Long Short-Term Memory)
- **Type:** Deep Learning
- **Architecture:** 2-layer stacked LSTM (64→32 units)
- **MAPE:** ~9.64%
- **Best for:** Capturing long-term sequential patterns

### 2. XGBoost with SHAP
- **Type:** Gradient Boosting Ensemble
- **Features:** 30+ engineered features
- **Explainability:** SHAP values for feature importance
- **Best for:** Understanding demand drivers

### 3. Temporal Fusion Transformer (TFT)
- **Type:** Attention-based Transformer
- **Components:** Self-attention + Positional encoding
- **Interpretability:** Temporal attention weights
- **Best for:** Complex multi-horizon forecasting

---

## 📁 Project Structure
- supply_chain_forecast/
- ├── app.py # Flask web application
- ├── requirements.txt # Python dependencies
- ├── templates/
- │ ├── index.html # Main forecasting page
- │ └── dashboard.html # Analytics dashboard
- ├── models/
- │ ├── lstm_model.py # LSTM model training
- │ ├── xgboost_model.py # XGBoost + SHAP
- │ └── tft_model.py # TFT model
- ├── utils/
- │ ├── config.py # Configuration
- │ ├── data_generator.py # Synthetic data generation
- │ └── preprocessing.py # Data preprocessing
- └── README.md

