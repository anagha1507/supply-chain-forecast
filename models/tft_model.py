import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

class TimeSeriesTransformer(nn.Module):
    """Simplified Temporal Fusion Transformer for Demand Forecasting"""
    
    def __init__(self, d_model=64, nhead=4, num_layers=3, dropout=0.1, seq_length=30):
        super(TimeSeriesTransformer, self).__init__()
        
        self.d_model = d_model
        self.seq_length = seq_length
        
        # Input projection
        self.input_projection = nn.Linear(1, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=seq_length)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Attention pooling
        self.attention_pooling = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(d_model, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(32)
        
    def forward(self, x):
        # x shape: (batch, seq_len, 1)
        batch_size = x.shape[0]
        
        # Project input to d_model dimensions
        x = self.input_projection(x)  # (batch, seq_len, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoder
        x = self.transformer_encoder(x)  # (batch, seq_len, d_model)
        
        # Layer norm
        x = self.layer_norm1(x)
        
        # Global attention pooling - create a query vector
        query = torch.mean(x, dim=1, keepdim=True)  # (batch, 1, d_model)
        attn_output, attn_weights = self.attention_pooling(query, x, x)
        x = attn_output.squeeze(1)  # (batch, d_model)
        
        # Dropout
        x = self.dropout(x)
        
        # FC layers with residual connections
        x = self.fc1(x)
        x = self.layer_norm2(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        
        return x
    
    def get_attention_weights(self, x):
        """Extract attention weights for interpretability"""
        batch_size = x.shape[0]
        
        # Project input
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        
        # Transformer encoder
        x = self.transformer_encoder(x)
        x = self.layer_norm1(x)
        
        # Get attention weights
        query = torch.mean(x, dim=1, keepdim=True)
        _, attn_weights = self.attention_pooling(query, x, x)
        
        return attn_weights


class PositionalEncoding(nn.Module):
    """Positional encoding for time series"""
    
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TFTModel:
    """Temporal Fusion Transformer Model Wrapper"""
    
    def __init__(self, config=None):
        self.config = config or {
            'd_model': 64,
            'nhead': 4,
            'num_layers': 3,
            'dropout': 0.1,
            'seq_length': 30,
            'epochs': 50,
            'batch_size': 32,
            'learning_rate': 0.001,
        }
        self.model = None
        self.train_losses = []
        self.val_losses = []
        self.attention_weights = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  Using device: {self.device}")
        
    def build_model(self):
        """Build the TFT model"""
        self.model = TimeSeriesTransformer(
            d_model=self.config['d_model'],
            nhead=self.config['nhead'],
            num_layers=self.config['num_layers'],
            dropout=self.config['dropout'],
            seq_length=self.config['seq_length']
        ).to(self.device)
        
        # Print model summary
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print("✅ TFT Model built successfully")
        print(f"📊 Model architecture:\n{self.model}")
        print(f"📊 Total parameters: {total_params:,}")
        print(f"📊 Trainable parameters: {trainable_params:,}")
        
    def train(self, X_train, y_train, X_val, y_val):
        """Train the TFT model"""
        print("\n🔄 Starting TFT training...")
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).to(self.device)
        
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
        criterion = nn.HuberLoss(delta=1.0)  # Robust to outliers
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=0.01
        )
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2
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
            scheduler.step(epoch + avg_val_loss / 1000)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'models/best_tft_model.pth')
            else:
                patience_counter += 1
            
            # Print progress
            if (epoch + 1) % 5 == 0:
                print(f"Epoch [{epoch+1}/{self.config['epochs']}] "
                      f"Train Loss: {avg_train_loss:.4f}, "
                      f"Val Loss: {avg_val_loss:.4f}")
            
            # Early stopping check
            if patience_counter >= 15:
                print(f"🛑 Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        self.model.load_state_dict(torch.load('models/best_tft_model.pth'))
        print("✅ TFT Training completed")
        
    def evaluate(self, X_test, y_test):
        """Evaluate the model"""
        print("\n📊 Evaluating TFT model...")
        
        self.model.eval()
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_test_tensor).cpu().numpy().flatten()
        
        # Calculate metrics
        mse = np.mean((predictions - y_test) ** 2)
        mae = np.mean(np.abs(predictions - y_test))
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_test - predictions) / (y_test + 1e-10))) * 100
        
        # R² score
        ss_res = np.sum((y_test - predictions) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        print(f"📈 TFT Test Metrics:")
        print(f"   MSE: {mse:.4f}")
        print(f"   MAE: {mae:.4f}")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   MAPE: {mape:.2f}%")
        print(f"   R² Score: {r2:.4f}")
        
        metrics = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'r2': r2
        }
        
        return predictions, metrics
    
    def extract_attention_weights(self, X_sample):
        """Extract attention weights for interpretability"""
        self.model.eval()
        X_tensor = torch.FloatTensor(X_sample).to(self.device)
        
        with torch.no_grad():
            attention = self.model.get_attention_weights(X_tensor)
        
        self.attention_weights = attention.cpu().numpy()
        return self.attention_weights
    
    def plot_attention_heatmap(self, X_sample, save_path='static/tft_attention.png'):
        """Plot attention weights as heatmap"""
        attention = self.extract_attention_weights(X_sample)
        
        # Average attention across heads
        avg_attention = attention.mean(axis=1).squeeze()
        
        plt.figure(figsize=(12, 4))
        plt.imshow(avg_attention.reshape(1, -1), aspect='auto', cmap='YlOrRd')
        plt.colorbar(label='Attention Weight')
        plt.xlabel('Time Step (Past Days)')
        plt.ylabel('Attention')
        plt.title('TFT: Temporal Attention Weights\n(Darker = More Important Time Steps)')
        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"📊 Attention heatmap saved to {save_path}")
    
    def plot_training_history(self):
        """Plot training and validation loss"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (Huber)')
        plt.title('TFT Training History')
        plt.legend()
        plt.grid(True)
        plt.savefig('static/tft_training_history.png')
        plt.close()
        print("📊 TFT Training history plot saved")
    
    def plot_predictions(self, y_true, y_pred, n_samples=100):
        """Plot actual vs predicted values"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Actual vs Predicted
        axes[0].plot(y_true[:n_samples], label='Actual', alpha=0.7, linewidth=1.5)
        axes[0].plot(y_pred[:n_samples], label='TFT Predicted', alpha=0.7, linewidth=1.5)
        axes[0].set_xlabel('Sample')
        axes[0].set_ylabel('Demand')
        axes[0].set_title('TFT: Actual vs Predicted Demand')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Scatter plot
        axes[1].scatter(y_true, y_pred, alpha=0.4, s=10)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        axes[1].set_xlabel('Actual Demand')
        axes[1].set_ylabel('Predicted Demand')
        axes[1].set_title('TFT: Prediction Scatter')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('static/tft_predictions.png', dpi=100, bbox_inches='tight')
        plt.close()
        print("📊 TFT Predictions plot saved")
    
    def predict_future(self, last_sequence, n_steps=7):
        """Predict future demand"""
        self.model.eval()
        
        predictions = []
        current_sequence = last_sequence.copy()
        
        with torch.no_grad():
            for _ in range(n_steps):
                X = torch.FloatTensor(current_sequence).reshape(1, -1, 1).to(self.device)
                pred = self.model(X).cpu().numpy()[0, 0]
                predictions.append(pred)
                
                current_sequence = np.roll(current_sequence, -1)
                current_sequence[-1] = pred
        
        return np.array(predictions)
    
    def save_model(self, path='models/tft_model.pkl'):
        """Save the TFT model"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        model_data = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }
        torch.save(model_data, path)
        print(f"💾 TFT model saved to {path}")
    
    def load_model(self, path='models/tft_model.pkl'):
        """Load a saved TFT model"""
        model_data = torch.load(path, map_location=self.device)
        self.config = model_data['config']
        self.build_model()
        self.model.load_state_dict(model_data['model_state_dict'])
        self.train_losses = model_data['train_losses']
        self.val_losses = model_data['val_losses']
        print(f"📂 TFT model loaded from {path}")


if __name__ == "__main__":
    import sys
    sys.path.append('.')
    from utils.preprocessing import DataPreprocessor
    
    # Load and prepare data
    preprocessor = DataPreprocessor()
    df = preprocessor.load_data()
    df = preprocessor.create_features(df)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.prepare_sequence_data(df)
    
    # Train TFT
    tft_model = TFTModel()
    tft_model.build_model()
    tft_model.train(X_train, y_train, X_val, y_val)
    
    # Evaluate
    predictions, metrics = tft_model.evaluate(X_test, y_test)
    
    # Plot results
    tft_model.plot_training_history()
    tft_model.plot_predictions(y_test, predictions)
    
    # Extract and plot attention weights
    X_sample = X_test[:10]
    tft_model.plot_attention_heatmap(X_sample)
    
    # Future prediction
    last_sequence = X_test[-1].flatten()
    future_preds = tft_model.predict_future(last_sequence, n_steps=7)
    print(f"\n🔮 TFT Future 7-day predictions: {future_preds}")
    
    # Save model
    tft_model.save_model()
    
    print("\n✅ TFT model training and evaluation complete!")