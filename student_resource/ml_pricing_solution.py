"""
ML Challenge 2025: Smart Product Pricing Solution
Comprehensive approach using both textual and visual features
"""

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer
import cv2
from PIL import Image
import requests
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

class ProductPricingPredictor:
    def __init__(self):
        self.text_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.models = {}
        
    def extract_text_features(self, catalog_content):
        """Extract structured features from catalog content"""
        features = {}
        
        # Extract item name
        item_name_match = re.search(r'Item Name: ([^\n]+)', catalog_content)
        features['item_name'] = item_name_match.group(1) if item_name_match else ''
        
        # Extract value and unit information
        value_match = re.search(r'Value: ([\d.]+)', catalog_content)
        features['value'] = float(value_match.group(1)) if value_match else 0.0
        
        unit_match = re.search(r'Unit: ([^\n]+)', catalog_content)
        features['unit'] = unit_match.group(1).strip() if unit_match else ''
        
        # Extract bullet points count
        bullet_points = re.findall(r'Bullet Point \d+:', catalog_content)
        features['bullet_points_count'] = len(bullet_points)
        
        # Extract product description
        desc_match = re.search(r'Product Description: ([^\n]+)', catalog_content)
        features['has_description'] = 1 if desc_match else 0
        
        # Text length features
        features['text_length'] = len(catalog_content)
        features['word_count'] = len(catalog_content.split())
        
        # Extract brand information (common patterns)
        brand_indicators = ['Brand:', 'by', 'Made by', 'Manufacturer:']
        features['has_brand_info'] = any(indicator in catalog_content for indicator in brand_indicators)
        
        # Extract package information
        package_indicators = ['Pack of', 'Pack of', 'Bundle', 'Set of', 'Multi-pack']
        features['is_packaged'] = any(indicator in catalog_content for indicator in package_indicators)
        
        # Extract size information
        size_indicators = ['oz', 'lb', 'g', 'kg', 'ml', 'L', 'inch', 'cm', 'mm']
        features['has_size_info'] = any(indicator in catalog_content.lower() for indicator in size_indicators)
        
        return features
    
    def extract_image_features(self, image_url, max_retries=3):
        """Extract basic image features"""
        features = {}
        
        try:
            for attempt in range(max_retries):
                try:
                    response = requests.get(image_url, timeout=10)
                    if response.status_code == 200:
                        image = Image.open(BytesIO(response.content))
                        
                        # Convert to RGB if necessary
                        if image.mode != 'RGB':
                            image = image.convert('RGB')
                        
                        # Basic image features
                        features['image_width'] = image.width
                        features['image_height'] = image.height
                        features['image_aspect_ratio'] = image.width / image.height if image.height > 0 else 0
                        features['image_area'] = image.width * image.height
                        
                        # Convert to numpy array for OpenCV
                        img_array = np.array(image)
                        
                        # Color features
                        features['mean_red'] = np.mean(img_array[:, :, 0])
                        features['mean_green'] = np.mean(img_array[:, :, 1])
                        features['mean_blue'] = np.mean(img_array[:, :, 2])
                        features['std_red'] = np.std(img_array[:, :, 0])
                        features['std_green'] = np.std(img_array[:, :, 1])
                        features['std_blue'] = np.std(img_array[:, :, 2])
                        
                        # Brightness and contrast
                        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                        features['brightness'] = np.mean(gray)
                        features['contrast'] = np.std(gray)
                        
                        break
                except Exception as e:
                    if attempt == max_retries - 1:
                        print(f"Failed to process image {image_url}: {e}")
                        
        except Exception as e:
            print(f"Error downloading image {image_url}: {e}")
        
        # Fill missing values with defaults
        default_features = {
            'image_width': 224, 'image_height': 224, 'image_aspect_ratio': 1.0,
            'image_area': 50176, 'mean_red': 128, 'mean_green': 128, 'mean_blue': 128,
            'std_red': 64, 'std_green': 64, 'std_blue': 64, 'brightness': 128, 'contrast': 64
        }
        
        for key, default_value in default_features.items():
            if key not in features:
                features[key] = default_value
                
        return features
    
    def calculate_smape(self, y_true, y_pred):
        """Calculate SMAPE metric"""
        return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)))
    
    def prepare_features(self, df, is_training=True, sample_size=None):
        """Prepare features for training or prediction"""
        print("Extracting text features...")
        
        # Sample data if specified (for faster development)
        if sample_size and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
        
        # Extract text features
        text_features = []
        for idx, catalog_content in enumerate(df['catalog_content']):
            if idx % 1000 == 0:
                print(f"Processing text features: {idx}/{len(df)}")
            text_features.append(self.extract_text_features(catalog_content))
        
        text_df = pd.DataFrame(text_features)
        
        print("Extracting image features...")
        # Extract image features (sample a subset for faster processing)
        image_sample_size = min(1000, len(df)) if not is_training else min(5000, len(df))
        image_features = []
        
        for idx, image_url in enumerate(df['image_link'].head(image_sample_size)):
            if idx % 100 == 0:
                print(f"Processing image features: {idx}/{image_sample_size}")
            image_features.append(self.extract_image_features(image_url))
        
        image_df = pd.DataFrame(image_features)
        
        # Combine features
        feature_df = pd.concat([text_df, image_df], axis=1)
        
        # Handle missing values
        feature_df = feature_df.fillna(0)
        
        # Encode categorical variables
        if is_training:
            for col in ['unit']:
                if col in feature_df.columns:
                    le = LabelEncoder()
                    feature_df[f'{col}_encoded'] = le.fit_transform(feature_df[col].astype(str))
                    self.label_encoders[col] = le
        else:
            for col in ['unit']:
                if col in feature_df.columns:
                    if col in self.label_encoders:
                        le = self.label_encoders[col]
                        feature_df[f'{col}_encoded'] = feature_df[col].astype(str).map(
                            lambda x: le.transform([x])[0] if x in le.classes_ else 0
                        ).fillna(0)
        
        # Drop original categorical columns
        feature_df = feature_df.drop(columns=['unit', 'item_name'], errors='ignore')
        
        return feature_df
    
    def train_models(self, X, y):
        """Train multiple models and select the best one"""
        print("Training models...")
        
        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Define models to try
        models = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'Ridge': Ridge(alpha=1.0),
            'LinearRegression': LinearRegression()
        }
        
        best_score = float('inf')
        best_model = None
        
        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X_train_scaled, y_train)
            
            # Predict on validation set
            y_pred = model.predict(X_val_scaled)
            
            # Calculate SMAPE
            smape = self.calculate_smape(y_val, y_pred)
            mae = mean_absolute_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)
            
            print(f"{name} - SMAPE: {smape:.2f}%, MAE: {mae:.2f}, R²: {r2:.4f}")
            
            self.models[name] = model
            
            if smape < best_score:
                best_score = smape
                best_model = name
        
        print(f"Best model: {best_model} with SMAPE: {best_score:.2f}%")
        return best_model
    
    def predict(self, X):
        """Make predictions using the best model"""
        X_scaled = self.scaler.transform(X)
        
        # Use ensemble of top models for better predictions
        predictions = []
        for name, model in self.models.items():
            if name in ['RandomForest', 'GradientBoosting']:  # Use ensemble of best models
                pred = model.predict(X_scaled)
                predictions.append(pred)
        
        # Average predictions
        final_predictions = np.mean(predictions, axis=0)
        
        # Ensure positive predictions
        final_predictions = np.maximum(final_predictions, 0.01)
        
        return final_predictions

def main():
    """Main training and prediction pipeline"""
    print("=== ML Challenge 2025: Smart Product Pricing Solution ===")
    
    # Initialize predictor
    predictor = ProductPricingPredictor()
    
    # Load data
    print("Loading training data...")
    train_df = pd.read_csv('dataset/train.csv')
    test_df = pd.read_csv('dataset/test.csv')
    
    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    
    # Sample data for faster development (remove this for full training)
    print("Using sample data for faster development...")
    train_sample = train_df.sample(n=min(10000, len(train_df)), random_state=42)
    
    # Prepare features
    print("Preparing features...")
    X_train = predictor.prepare_features(train_sample, is_training=True)
    y_train = train_sample['price']
    
    print(f"Feature matrix shape: {X_train.shape}")
    print(f"Features: {list(X_train.columns)}")
    
    # Train models
    best_model = predictor.train_models(X_train, y_train)
    
    # Prepare test features
    print("Preparing test features...")
    X_test = predictor.prepare_features(test_df, is_training=False)
    
    # Make predictions
    print("Making predictions...")
    predictions = predictor.predict(X_test)
    
    # Create submission file
    submission_df = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': predictions
    })
    
    # Save predictions
    output_file = 'dataset/test_out.csv'
    submission_df.to_csv(output_file, index=False)
    
    print(f"Predictions saved to {output_file}")
    print(f"Total predictions: {len(submission_df)}")
    print(f"Sample predictions:")
    print(submission_df.head())
    
    # Display statistics
    print(f"\nPrediction Statistics:")
    print(f"Mean price: ${predictions.mean():.2f}")
    print(f"Median price: ${np.median(predictions):.2f}")
    print(f"Min price: ${predictions.min():.2f}")
    print(f"Max price: ${predictions.max():.2f}")
    
    return predictor, submission_df

if __name__ == "__main__":
    predictor, submission = main()
