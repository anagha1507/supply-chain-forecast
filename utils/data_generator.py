import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

class SupplyChainDataGenerator:
    def __init__(self):
        np.random.seed(42)
        
    def generate_data(self, n_products=10, n_stores=5, n_days=365):
        """Generate synthetic supply chain data"""
        
        print("🔄 Generating supply chain data...")
        
        # Create date range
        dates = pd.date_range(
            start='2023-01-01', 
            periods=n_days, 
            freq='D'
        )
        
        data = []
        
        for product_id in range(1, n_products + 1):
            for store_id in range(1, n_stores + 1):
                for date in dates:
                    # Create base demand with seasonality
                    day_of_week = date.dayofweek
                    month = date.month
                    day_of_year = date.dayofyear
                    
                    # Seasonal patterns
                    weekly_pattern = 1 + 0.2 * np.sin(2 * np.pi * day_of_week / 7)
                    monthly_pattern = 1 + 0.3 * np.sin(2 * np.pi * month / 12)
                    yearly_pattern = 1 + 0.15 * np.sin(2 * np.pi * day_of_year / 365)
                    
                    # Product and store specific factors
                    product_factor = 50 + (product_id * 20)
                    store_factor = 0.8 + (store_id * 0.1)
                    
                    # Base demand calculation
                    base_demand = (
                        product_factor * 
                        store_factor * 
                        weekly_pattern * 
                        monthly_pattern * 
                        yearly_pattern
                    )
                    
                    # Add noise
                    noise = np.random.normal(0, base_demand * 0.1)
                    demand = max(0, base_demand + noise)
                    
                    # Calculate derived features
                    inventory_level = demand * np.random.uniform(1.1, 1.5)
                    lead_time = np.random.randint(1, 7)
                    stockout_risk = 0 if inventory_level > demand * 1.2 else np.random.uniform(0.3, 0.9)
                    
                    # Price and promotions
                    base_price = product_factor * 0.5
                    price = base_price * np.random.uniform(0.8, 1.2)
                    is_promotion = np.random.choice([0, 1], p=[0.85, 0.15])
                    discount_pct = np.random.uniform(0.1, 0.3) if is_promotion else 0
                    
                    # Additional features
                    holiday = 1 if date.month in [11, 12] and date.day in [24, 25, 26, 31] else 0
                    
                    data.append({
                        'date': date,
                        'product_id': product_id,
                        'store_id': store_id,
                        'demand': round(demand, 2),
                        'inventory_level': round(inventory_level, 2),
                        'lead_time': lead_time,
                        'stockout_risk': round(stockout_risk, 3),
                        'price': round(price, 2),
                        'is_promotion': is_promotion,
                        'discount_pct': round(discount_pct, 3),
                        'day_of_week': day_of_week,
                        'month': month,
                        'holiday': holiday,
                        'weekend': 1 if day_of_week >= 5 else 0
                    })
        
        df = pd.DataFrame(data)
        print(f"✅ Generated {len(df)} records")
        return df
    
    def save_data(self, df, path='data/supply_chain_data.csv'):
        """Save generated data to CSV"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        print(f"💾 Data saved to {path}")

if __name__ == "__main__":
    generator = SupplyChainDataGenerator()
    df = generator.generate_data(n_products=10, n_stores=5, n_days=365)
    generator.save_data(df)
    print("\n📊 Data Preview:")
    print(df.head())
    print(f"\n📈 Data Shape: {df.shape}")
    print(f"\n📋 Columns: {df.columns.tolist()}")