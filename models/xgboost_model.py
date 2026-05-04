import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class XGBoostModel:
    def __init__(self, config=None):
        self.config = config or {
            'n_estimators': 200,
            'max_depth': 7,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'early_stopping_rounds': 20,
        }
        self.model = None
        self.feature_importance = None
        self.shap_values = None
        self.shap_explainer = None
        
    def build_model(self):
        """Initialize XGBoost model with early stopping capability"""
        self.model = xgb.XGBRegressor(
            n_estimators=self.config['n_estimators'],
            max_depth=self.config['max_depth'],
            learning_rate=self.config['learning_rate'],
            subsample=self.config['subsample'],
            colsample_bytree=self.config['colsample_bytree'],
            min_child_weight=self.config['min_child_weight'],
            gamma=self.config['gamma'],
            reg_alpha=self.config['reg_alpha'],
            reg_lambda=self.config['reg_lambda'],
            random_state=42,
            n_jobs=-1,
            verbosity=1,
        )
        print("✅ XGBoost model built successfully")
        
    def train(self, X_train, y_train, X_val, y_val):
        """Train XGBoost model with early stopping"""
        print("\n🔄 Starting XGBoost training...")
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False
        )
        
        print("✅ XGBoost training completed")
        
    def evaluate(self, X_test, y_test):
        """Evaluate the model"""
        print("\n📊 Evaluating XGBoost model...")
        
        predictions = self.model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predictions)
        mape = np.mean(np.abs((y_test - predictions) / (y_test + 1e-10))) * 100
        
        print(f"📈 XGBoost Test Metrics:")
        print(f"   MSE: {mse:.4f}")
        print(f"   MAE: {mae:.4f}")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   R² Score: {r2:.4f}")
        print(f"   MAPE: {mape:.2f}%")
        
        metrics = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape
        }
        
        return predictions, metrics
    
    def get_feature_importance(self, feature_names):
        """Get feature importance from XGBoost"""
        importance = self.model.feature_importances_
        
        # Create DataFrame
        self.feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return self.feature_importance
    
    def plot_feature_importance(self, top_n=20):
        """Plot feature importance"""
        if self.feature_importance is None:
            print("❌ Run get_feature_importance() first")
            return
        
        plt.figure(figsize=(12, 8))
        
        # Get top N features
        top_features = self.feature_importance.head(top_n)
        
        # Create bar plot
        plt.barh(range(len(top_features)), top_features['importance'].values)
        plt.yticks(range(len(top_features)), top_features['feature'].values)
        plt.xlabel('Importance')
        plt.title(f'XGBoost Feature Importance (Top {top_n})')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('static/xgboost_feature_importance.png', dpi=100, bbox_inches='tight')
        plt.close()
        print("📊 Feature importance plot saved to static/xgboost_feature_importance.png")
    
    def calculate_shap(self, X_sample, feature_names):
        """Calculate SHAP values for model explainability"""
        print("\n🔄 Calculating SHAP values...")
        
        # Create SHAP explainer
        self.shap_explainer = shap.TreeExplainer(self.model)
        
        # Calculate SHAP values
        self.shap_values = self.shap_explainer.shap_values(X_sample)
        
        # Store feature names
        self.shap_feature_names = feature_names
        
        print("✅ SHAP values calculated")
        return self.shap_values
    
    def plot_shap_summary(self):
        """Create SHAP summary plot"""
        if self.shap_values is None:
            print("❌ Run calculate_shap() first")
            return
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            self.shap_values, 
            feature_names=self.shap_feature_names,
            show=False,
            max_display=20
        )
        plt.tight_layout()
        plt.savefig('static/shap_summary.png', dpi=100, bbox_inches='tight')
        plt.close()
        print("📊 SHAP summary plot saved to static/shap_summary.png")
    
    def plot_shap_dependence(self, feature_name):
        """Create SHAP dependence plot for a specific feature"""
        if self.shap_values is None:
            print("❌ Run calculate_shap() first")
            return
        
        feature_idx = self.shap_feature_names.index(feature_name)
        
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            feature_idx,
            self.shap_values,
            feature_names=self.shap_feature_names,
            show=False
        )
        plt.title(f'SHAP Dependence Plot - {feature_name}')
        plt.tight_layout()
        plt.savefig(f'static/shap_dependence_{feature_name}.png', dpi=100, bbox_inches='tight')
        plt.close()
        print(f"📊 SHAP dependence plot for {feature_name} saved")
    
    def plot_shap_waterfall(self, X_single, feature_names):
        """Create SHAP waterfall plot for a single prediction"""
        if self.shap_explainer is None:
            print("❌ Run calculate_shap() first")
            return
        
        # Calculate SHAP for single instance
        shap_values_single = self.shap_explainer(X_single)
        
        plt.figure(figsize=(12, 8))
        shap.waterfall_plot(
            shap_values_single[0],
            show=False,
            max_display=10
        )
        plt.title('SHAP Waterfall Plot - Single Prediction Explanation')
        plt.tight_layout()
        plt.savefig('static/shap_waterfall.png', dpi=100, bbox_inches='tight')
        plt.close()
        print("📊 SHAP waterfall plot saved to static/shap_waterfall.png")
    
    def get_shap_feature_importance(self):
        """Get mean absolute SHAP values for feature importance"""
        shap_importance = np.abs(self.shap_values).mean(axis=0)
        
        shap_importance_df = pd.DataFrame({
            'feature': self.shap_feature_names,
            'shap_importance': shap_importance
        }).sort_values('shap_importance', ascending=False)
        
        return shap_importance_df
    
    def plot_actual_vs_predicted(self, y_true, y_pred):
        """Plot actual vs predicted values"""
        plt.figure(figsize=(12, 5))
        
        # Scatter plot
        plt.subplot(1, 2, 1)
        plt.scatter(y_true, y_pred, alpha=0.5)
        
        # Add diagonal line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        
        plt.xlabel('Actual Demand')
        plt.ylabel('Predicted Demand')
        plt.title('XGBoost: Actual vs Predicted')
        
        # Residuals plot
        plt.subplot(1, 2, 2)
        residuals = y_true - y_pred
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Demand')
        plt.ylabel('Residuals')
        plt.title('Residuals Plot')
        
        plt.tight_layout()
        plt.savefig('static/xgboost_predictions.png', dpi=100, bbox_inches='tight')
        plt.close()
        print("📊 Predictions plot saved to static/xgboost_predictions.png")
    
    def save_model(self, path='models/xgboost_model.pkl'):
        """Save the trained model"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        print(f"💾 XGBoost model saved to {path}")
    
    def load_model(self, path='models/xgboost_model.pkl'):
        """Load a saved model"""
        self.model = joblib.load(path)
        print(f"📂 XGBoost model loaded from {path}")


if __name__ == "__main__":
    # Test XGBoost model
    import sys
    sys.path.append('.')
    from utils.preprocessing import DataPreprocessor
    
    # Load and prepare data
    preprocessor = DataPreprocessor()
    df = preprocessor.load_data()
    df = preprocessor.create_features(df)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.prepare_tabular_data(df)
    
    # Get feature names
    feature_names = preprocessor.feature_columns
    
    # Train XGBoost
    xgb_model = XGBoostModel()
    xgb_model.build_model()
    xgb_model.train(X_train, y_train, X_val, y_val)
    
    # Evaluate
    predictions, metrics = xgb_model.evaluate(X_test, y_test)
    
    # Feature importance
    xgb_model.get_feature_importance(feature_names)
    xgb_model.plot_feature_importance(top_n=20)
    
    # SHAP analysis
    # Use a subset of test data for SHAP (faster computation)
    X_sample = X_test.iloc[:500].values
    xgb_model.calculate_shap(X_sample, feature_names)
    xgb_model.plot_shap_summary()
    
    # Plot predictions
    xgb_model.plot_actual_vs_predicted(y_test.values, predictions)
    
    # Save model
    xgb_model.save_model()
    
    print("\n✅ XGBoost model training, evaluation, and SHAP analysis complete!")