import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib
import os
import matplotlib.pyplot as plt
from datetime import datetime

class LSTMForecaster(nn.Module):
    """LSTM Model for Time Series Demand Forecasting"""
    
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2, output_size=1):
        super(LSTMForecaster, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Stacked LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Pass through stacked LSTM
        lstm_out, _ = self.lstm(x)
        
        # Take the last time step output
        x = lstm_out[:, -1, :]
        
        # Dropout
        x = self.dropout(x)
        
        # Pass through FC layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

class LSTMModel:
    def __init__(self, config=None):
        self.config = config or {
            'sequence_length': 30,
            'n_features': 1,
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.2,
            'epochs': 50,
            'batch_size': 32,
            'learning_rate': 0.001,
        }
        self.model = None
        self.train_losses = []
        self.val_losses = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  Using device: {self.device}")
        
    def build_model(self):
        """Build the LSTM model"""
        self.model = LSTMForecaster(
            input_size=self.config['n_features'],
            hidden_size=self.config['hidden_size'],
            num_layers=self.config['num_layers'],
            dropout=self.config['dropout'],
            output_size=1
        ).to(self.device)
        
        # Print model summary
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print("✅ LSTM Model built successfully")
        print(f"📊 Model architecture:\n{self.model}")
        print(f"📊 Total parameters: {total_params:,}")
        print(f"📊 Trainable parameters: {trainable_params:,}")
        
    def train(self, X_train, y_train, X_val, y_val):
        """Train the LSTM model"""
        print("\n🔄 Starting LSTM training...")
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).to(self.device)
        
        # Print shapes for debugging
        print(f"📐 Training data shape: {X_train_tensor.shape}")
        print(f"📐 Validation data shape: {X_val_tensor.shape}")
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=False
        )
        
        # Define loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5
        )
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config['epochs']):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            self.train_losses.append(avg_train_loss)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = self.model(batch_X)
                    loss = criterion(outputs.squeeze(), batch_y)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            self.val_losses.append(avg_val_loss)
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'models/best_lstm_model.pth')
            else:
                patience_counter += 1
            
            # Print progress
            if (epoch + 1) % 5 == 0:
                print(f"Epoch [{epoch+1}/{self.config['epochs']}] "
                      f"Train Loss: {avg_train_loss:.4f}, "
                      f"Val Loss: {avg_val_loss:.4f}")
            
            # Early stopping check
            if patience_counter >= 10:
                print(f"🛑 Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        self.model.load_state_dict(torch.load('models/best_lstm_model.pth'))
        print("✅ LSTM Training completed")
        
    def evaluate(self, X_test, y_test):
        """Evaluate the model"""
        print("\n📊 Evaluating LSTM model...")
        
        self.model.eval()
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_test_tensor).cpu().numpy().flatten()
        
        # Calculate metrics
        mse = np.mean((predictions - y_test) ** 2)
        mae = np.mean(np.abs(predictions - y_test))
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_test - predictions) / (y_test + 1e-10))) * 100
        
        print(f"📈 LSTM Test Metrics:")
        print(f"   MSE: {mse:.4f}")
        print(f"   MAE: {mae:.4f}")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   MAPE: {mape:.2f}%")
        
        metrics = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'mape': mape
        }
        
        return predictions, metrics
    
    def predict_future(self, last_sequence, n_steps=7):
        """Predict future demand for n steps"""
        self.model.eval()
        
        predictions = []
        current_sequence = last_sequence.copy()
        
        with torch.no_grad():
            for _ in range(n_steps):
                # Prepare input - reshape to (1, sequence_length, n_features)
                X = torch.FloatTensor(current_sequence).reshape(1, -1, 1).to(self.device)
                
                # Make prediction
                pred = self.model(X).cpu().numpy()[0, 0]
                predictions.append(pred)
                
                # Update sequence (remove first element, add prediction)
                current_sequence = np.roll(current_sequence, -1)
                current_sequence[-1] = pred
        
        return np.array(predictions)
    
    def plot_training_history(self):
        """Plot training and validation loss"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('LSTM Training History')
        plt.legend()
        plt.grid(True)
        plt.savefig('static/lstm_training_history.png')
        plt.close()
        print("📊 Training history plot saved to static/lstm_training_history.png")
    
    def plot_predictions(self, y_true, y_pred, n_samples=100):
        """Plot actual vs predicted values"""
        plt.figure(figsize=(12, 6))
        plt.plot(y_true[:n_samples], label='Actual', alpha=0.7)
        plt.plot(y_pred[:n_samples], label='Predicted', alpha=0.7)
        plt.xlabel('Sample')
        plt.ylabel('Demand')
        plt.title('LSTM: Actual vs Predicted Demand')
        plt.legend()
        plt.grid(True)
        plt.savefig('static/lstm_predictions.png')
        plt.close()
        print("📊 Predictions plot saved to static/lstm_predictions.png")
    
    def save_model(self, path='models/lstm_model.pkl'):
        """Save the model and config"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        model_data = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }
        torch.save(model_data, path)
        print(f"💾 LSTM model saved to {path}")
    
    def load_model(self, path='models/lstm_model.pkl'):
        """Load a saved model"""
        model_data = torch.load(path, map_location=self.device)
        self.config = model_data['config']
        self.build_model()
        self.model.load_state_dict(model_data['model_state_dict'])
        self.train_losses = model_data['train_losses']
        self.val_losses = model_data['val_losses']
        print(f"📂 LSTM model loaded from {path}")


if __name__ == "__main__":
    # Test LSTM model
    import sys
    sys.path.append('.')
    from utils.preprocessing import DataPreprocessor
    
    # Load and prepare data
    preprocessor = DataPreprocessor()
    df = preprocessor.load_data()
    df = preprocessor.create_features(df)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.prepare_sequence_data(df)
    
    # Train LSTM
    lstm_model = LSTMModel()
    lstm_model.build_model()
    lstm_model.train(X_train, y_train, X_val, y_val)
    
    # Evaluate
    predictions, metrics = lstm_model.evaluate(X_test, y_test)
    
    # Plot results
    lstm_model.plot_training_history()
    lstm_model.plot_predictions(y_test, predictions)
    
    # Test future prediction
    last_sequence = X_test[-1].flatten()
    future_preds = lstm_model.predict_future(last_sequence, n_steps=7)
    print(f"\n🔮 Future 7-day predictions: {future_preds}")
    
    # Save model
    lstm_model.save_model()
    
    print("\n✅ LSTM model training and evaluation complete!")