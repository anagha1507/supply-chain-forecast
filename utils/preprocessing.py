import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
import os
from datetime import datetime

class DataPreprocessor:
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_columns = []
        self.target_column = 'demand'
        
    def load_data(self, path='data/supply_chain_data.csv'):
        """Load the supply chain data"""
        print("📂 Loading data...")
        df = pd.read_csv(path)
        df['date'] = pd.to_datetime(df['date'])
        print(f"✅ Loaded {len(df)} records")
        return df
    
    def create_features(self, df):
        """Create additional features for better forecasting"""
        print("🔧 Creating features...")
        
        # Sort data
        df = df.sort_values(['product_id', 'store_id', 'date']).reset_index(drop=True)
        
        # Lag features (previous days demand)
        for lag in [1, 2, 3, 7, 14, 30]:
            df[f'demand_lag_{lag}'] = df.groupby(['product_id', 'store_id'])['demand'].shift(lag)
        
        # Rolling statistics
        for window in [3, 7, 14, 30]:
            df[f'demand_rolling_mean_{window}'] = df.groupby(['product_id', 'store_id'])['demand'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            df[f'demand_rolling_std_{window}'] = df.groupby(['product_id', 'store_id'])['demand'].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )
        
        # Price features
        df['price_change'] = df.groupby(['product_id', 'store_id'])['price'].diff()
        df['price_change_pct'] = df.groupby(['product_id', 'store_id'])['price'].pct_change()
        
        # Inventory features
        df['inventory_to_demand_ratio'] = df['inventory_level'] / (df['demand'] + 1)
        
        # Date features
        df['day_of_month'] = df['date'].dt.day
        df['quarter'] = df['date'].dt.quarter
        df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
        
        # Drop rows with NaN (from lag features)
        df = df.dropna()
        
        print(f"✅ Features created. Shape: {df.shape}")
        return df
    
    def prepare_tabular_data(self, df, test_size=0.2, val_size=0.2):
        """Prepare data for traditional ML models (XGBoost, etc.)"""
        print("📊 Preparing tabular data...")
        
        # Select features
        categorical_cols = ['product_id', 'store_id', 'day_of_week', 'month', 
                           'is_promotion', 'holiday', 'weekend', 'quarter']
        numerical_cols = ['inventory_level', 'lead_time', 'stockout_risk', 
                         'price', 'discount_pct'] + \
                        [col for col in df.columns if 'lag' in col or 'rolling' in col] + \
                        ['price_change', 'price_change_pct', 'inventory_to_demand_ratio',
                         'day_of_month', 'week_of_year']
        
        # Encode categorical variables
        for col in categorical_cols:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                df[f'{col}_encoded'] = self.encoders[col].fit_transform(df[col])
        
        encoded_cat_cols = [f'{col}_encoded' for col in categorical_cols]
        self.feature_columns = numerical_cols + encoded_cat_cols
        
        # Prepare X and y
        X = df[self.feature_columns].copy()
        y = df[self.target_column].copy()
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size/(1-test_size), random_state=42
        )
        
        # Scale numerical features
        self.scalers['numerical'] = StandardScaler()
        X_train[numerical_cols] = self.scalers['numerical'].fit_transform(X_train[numerical_cols])
        X_val[numerical_cols] = self.scalers['numerical'].transform(X_val[numerical_cols])
        X_test[numerical_cols] = self.scalers['numerical'].transform(X_test[numerical_cols])
        
        print(f"✅ Data prepared. Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def prepare_sequence_data(self, df, sequence_length=30, test_size=0.2):
        """Prepare data for LSTM and deep learning models"""
        print("📊 Preparing sequence data...")
        
        sequences = []
        targets = []
        
        # Group by product and store
        for (product, store), group in df.groupby(['product_id', 'store_id']):
            group = group.sort_values('date')
            
            # Use demand values only (univariate for simplicity)
            demand_values = group['demand'].values
            
            # Create sequences
            for i in range(len(demand_values) - sequence_length):
                seq = demand_values[i:i+sequence_length]
                target = demand_values[i+sequence_length]
                sequences.append(seq)
                targets.append(target)
        
        X = np.array(sequences).reshape(-1, sequence_length, 1)
        y = np.array(targets)
        
        # Train-test split
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Further split train into train and validation
        val_idx = int(len(X_train) * 0.8)
        X_val = X_train[val_idx:]
        y_val = y_train[val_idx:]
        X_train = X_train[:val_idx]
        y_train = y_train[:val_idx]
        
        # Scale data
        self.scalers['sequence'] = MinMaxScaler()
        X_train_reshaped = X_train.reshape(-1, 1)
        X_train_scaled = self.scalers['sequence'].fit_transform(X_train_reshaped)
        X_train = X_train_scaled.reshape(X_train.shape)
        
        X_val_reshaped = X_val.reshape(-1, 1)
        X_val_scaled = self.scalers['sequence'].transform(X_val_reshaped)
        X_val = X_val_scaled.reshape(X_val.shape)
        
        X_test_reshaped = X_test.reshape(-1, 1)
        X_test_scaled = self.scalers['sequence'].transform(X_test_reshaped)
        X_test = X_test_scaled.reshape(X_test.shape)
        
        print(f"✅ Sequence data prepared. Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def save_preprocessors(self, path='models'):
        """Save scalers and encoders"""
        os.makedirs(path, exist_ok=True)
        joblib.dump(self.scalers, os.path.join(path, 'scalers.pkl'))
        joblib.dump(self.encoders, os.path.join(path, 'encoders.pkl'))
        joblib.dump(self.feature_columns, os.path.join(path, 'feature_columns.pkl'))
        print(f"💾 Preprocessors saved to {path}")

if __name__ == "__main__":
    # Test the preprocessing pipeline
    preprocessor = DataPreprocessor()
    df = preprocessor.load_data()
    df = preprocessor.create_features(df)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.prepare_tabular_data(df)
    X_seq_train, X_seq_val, X_seq_test, y_seq_train, y_seq_val, y_seq_test = preprocessor.prepare_sequence_data(df)
    preprocessor.save_preprocessors()
    print("\n✅ Preprocessing pipeline test complete!")