"""
Fast CatBoost ML Challenge 2025: Smart Product Pricing Solution
Target: 20-30% SMAPE using optimized CatBoost with advanced features
Ultra-fast execution with high performance
"""

import os
import re
import pandas as pd
import numpy as np
import time
from catboost import CatBoostRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class FastCatBoostPricingPredictor:
    def __init__(self):
        self.catboost_model = None
        self.text_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 3))
        self.label_encoders = {}
        
    def extract_advanced_text_features(self, catalog_content):
        """Extract comprehensive features for optimal CatBoost performance"""
        features = {}
        
        # Enhanced text statistics
        features['text_length'] = len(catalog_content)
        features['word_count'] = len(catalog_content.split())
        features['sentence_count'] = len(re.findall(r'[.!?]+', catalog_content))
        features['avg_word_length'] = np.mean([len(word) for word in catalog_content.split()]) if catalog_content.split() else 0
        features['unique_word_ratio'] = len(set(catalog_content.lower().split())) / max(len(catalog_content.split()), 1)
        features['caps_ratio'] = sum(1 for c in catalog_content if c.isupper()) / max(len(catalog_content), 1)
        features['digit_ratio'] = sum(1 for c in catalog_content if c.isdigit()) / max(len(catalog_content), 1)
        
        # Extract structured data with improved patterns
        value_match = re.search(r'Value: ([\d.]+)', catalog_content)
        features['value'] = float(value_match.group(1)) if value_match else 0.0
        
        unit_match = re.search(r'Unit: ([^\n]+)', catalog_content)
        features['unit'] = unit_match.group(1).strip() if unit_match else ''
        
        # Enhanced bullet points analysis
        bullet_points = re.findall(r'Bullet Point \d+:', catalog_content)
        features['bullet_points_count'] = len(bullet_points)
        
        # Extract bullet point content
        bullet_content = re.findall(r'Bullet Point \d+: ([^\n]+)', catalog_content)
        features['bullet_text_length'] = sum(len(bp) for bp in bullet_content)
        features['avg_bullet_length'] = np.mean([len(bp) for bp in bullet_content]) if bullet_content else 0
        
        # Package information with better extraction
        pack_match = re.search(r'Pack of (\d+)', catalog_content)
        features['pack_size'] = int(pack_match.group(1)) if pack_match else 1
        
        case_match = re.search(r'Case of (\d+)', catalog_content)
        features['case_size'] = int(case_match.group(1)) if case_match else 1
        
        # Size extraction with values
        size_patterns = {
            'oz': r'(\d+(?:\.\d+)?)\s*oz',
            'lb': r'(\d+(?:\.\d+)?)\s*lb',
            'g': r'(\d+(?:\.\d+)?)\s*g',
            'kg': r'(\d+(?:\.\d+)?)\s*kg',
            'ml': r'(\d+(?:\.\d+)?)\s*ml',
            'L': r'(\d+(?:\.\d+)?)\s*L'
        }
        
        features['has_size_info'] = 0
        features['size_value'] = 0
        features['size_unit'] = ''
        
        for unit, pattern in size_patterns.items():
            match = re.search(pattern, catalog_content, re.IGNORECASE)
            if match:
                features['has_size_info'] = 1
                features['size_value'] = float(match.group(1))
                features['size_unit'] = unit
                break
        
        # Enhanced indicators
        brand_indicators = ['Brand:', 'by', 'Made by', 'Manufacturer:', 'Produced by', 'Distributed by']
        features['has_brand_info'] = any(indicator in catalog_content for indicator in brand_indicators)
        
        package_indicators = ['Pack of', 'Bundle', 'Set of', 'Multi-pack', 'Combo', 'Kit', 'Case of', 'Bulk']
        features['is_packaged'] = any(indicator in catalog_content for indicator in package_indicators)
        
        quality_indicators = ['Premium', 'Deluxe', 'Professional', 'High Quality', 'Luxury', 'Gourmet', 'Artisan', 'Craft', 'Organic']
        features['has_quality_indicators'] = any(indicator in catalog_content for indicator in quality_indicators)
        
        safety_indicators = ['Organic', 'Natural', 'Non-GMO', 'Gluten Free', 'Vegan', 'Kosher', 'Halal', 'Sugar Free', 'Low Fat', 'No Preservatives']
        features['has_safety_indicators'] = any(indicator in catalog_content for indicator in safety_indicators)
        
        price_indicators = ['Sale', 'Discount', 'Special', 'Limited', 'Exclusive', 'Rare', 'Clearance', 'Bulk']
        features['has_price_indicators'] = any(indicator in catalog_content for indicator in price_indicators)
        
        # Comprehensive numeric analysis
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', catalog_content)
        if numbers:
            numeric_values = [float(n) for n in numbers]
            features['numeric_count'] = len(numeric_values)
            features['max_number'] = max(numeric_values)
            features['min_number'] = min(numeric_values)
            features['avg_number'] = np.mean(numeric_values)
            features['median_number'] = np.median(numeric_values)
            features['std_number'] = np.std(numeric_values)
            
            # Large numbers (likely prices or quantities)
            large_numbers = [n for n in numeric_values if n > 10]
            features['large_number_count'] = len(large_numbers)
            features['large_number_avg'] = np.mean(large_numbers) if large_numbers else 0
            features['large_number_max'] = max(large_numbers) if large_numbers else 0
            
            # Very large numbers (bulk items)
            very_large = [n for n in numeric_values if n > 100]
            features['very_large_count'] = len(very_large)
            features['very_large_avg'] = np.mean(very_large) if very_large else 0
        else:
            features.update({
                'numeric_count': 0, 'max_number': 0, 'min_number': 0, 'avg_number': 0,
                'median_number': 0, 'std_number': 0, 'large_number_count': 0, 'large_number_avg': 0,
                'large_number_max': 0, 'very_large_count': 0, 'very_large_avg': 0
            })
        
        # Enhanced category detection
        category_keywords = {
            'food': ['food', 'snack', 'candy', 'chocolate', 'sauce', 'spice', 'beverage', 'drink', 'tea', 'coffee', 'cereal', 'pasta', 'rice', 'cookies', 'crackers'],
            'electronics': ['electronic', 'digital', 'battery', 'cable', 'charger', 'device', 'phone', 'tablet', 'laptop', 'computer'],
            'beauty': ['beauty', 'cosmetic', 'makeup', 'skincare', 'shampoo', 'lotion', 'cream', 'serum', 'fragrance'],
            'home': ['kitchen', 'cookware', 'furniture', 'decor', 'cleaning', 'storage', 'organizer', 'appliance'],
            'health': ['vitamin', 'supplement', 'medicine', 'health', 'fitness', 'medical', 'protein', 'wellness'],
            'clothing': ['shirt', 'pants', 'dress', 'jacket', 'shoes', 'clothing', 'apparel', 'fashion']
        }
        
        for category, keywords in category_keywords.items():
            features[f'is_{category}'] = any(keyword in catalog_content.lower() for keyword in keywords)
        
        # Price prediction features
        features['value_to_word_ratio'] = features['value'] / max(features['word_count'], 1)
        features['value_to_text_ratio'] = features['value'] / max(features['text_length'], 1)
        features['pack_size_to_value_ratio'] = features['pack_size'] / max(features['value'], 1)
        features['size_value_to_value_ratio'] = features['size_value'] / max(features['value'], 1)
        
        # Text complexity features
        features['exclamation_count'] = catalog_content.count('!')
        features['question_count'] = catalog_content.count('?')
        features['colon_count'] = catalog_content.count(':')
        features['semicolon_count'] = catalog_content.count(';')
        
        return features
    
    def calculate_smape(self, y_true, y_pred):
        """Calculate SMAPE metric"""
        return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)))
    
    def prepare_features(self, df, is_training=True, sample_size=None):
        """Prepare comprehensive features for CatBoost"""
        print("Preparing advanced features...")
        
        if sample_size and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
        
        # Extract advanced features
        features_list = []
        for idx, catalog_content in enumerate(df['catalog_content']):
            if idx % 2000 == 0 and idx > 0:
                print(f"Processing features: {idx}/{len(df)}")
            features_list.append(self.extract_advanced_text_features(catalog_content))
        
        feature_df = pd.DataFrame(features_list)
        
        # Handle missing values
        feature_df = feature_df.fillna(0)
        
        # Enhanced categorical encoding
        categorical_columns = ['unit', 'size_unit']
        for col in categorical_columns:
            if col in feature_df.columns:
                if is_training:
                    le = LabelEncoder()
                    feature_df[f'{col}_encoded'] = le.fit_transform(feature_df[col].astype(str))
                    self.label_encoders[col] = le
                else:
                    if col in self.label_encoders:
                        le = self.label_encoders[col]
                        feature_df[f'{col}_encoded'] = feature_df[col].astype(str).map(
                            lambda x: le.transform([x])[0] if x in le.classes_ else 0
                        ).fillna(0)
        
        # Add TF-IDF features
        if is_training:
            text_features = self.text_vectorizer.fit_transform(df['catalog_content'])
            self.text_feature_names = [f'text_{i}' for i in range(text_features.shape[1])]
        else:
            text_features = self.text_vectorizer.transform(df['catalog_content'])
        
        text_df = pd.DataFrame(text_features.toarray(), columns=self.text_feature_names)
        
        # Combine all features
        final_features = pd.concat([feature_df, text_df], axis=1)
        
        # Drop original categorical columns
        final_features = final_features.drop(columns=['unit', 'size_unit'], errors='ignore')
        
        return final_features
    
    def train_catboost_model(self, X, y):
        """Train optimized CatBoost model"""
        print("Training optimized CatBoost model...")
        
        # Remove extreme outliers (above 99th percentile)
        price_99th = np.percentile(y, 99)
        price_1st = np.percentile(y, 1)
        mask = (y >= price_1st) & (y <= price_99th)
        X_clean = X[mask]
        y_clean = y[mask]
        
        print(f"Training on {len(X_clean)} samples after outlier removal")
        print(f"Price range: ${y_clean.min():.2f} - ${y_clean.max():.2f}")
        
        # Apply log transformation for better distribution
        y_log = np.log1p(y_clean)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(X_clean, y_log, test_size=0.2, random_state=42)
        
        # Optimized CatBoost parameters
        self.catboost_model = CatBoostRegressor(
            iterations=2000,
            learning_rate=0.05,
            depth=10,
            l2_leaf_reg=3,
            loss_function='RMSE',
            random_seed=42,
            verbose=False,
            early_stopping_rounds=100,
            bootstrap_type='Bayesian',
            bagging_temperature=1,
            random_strength=1,
            rsm=0.8,
            min_data_in_leaf=20,
            max_leaves=31,
            grow_policy='Lossguide'
        )
        
        self.catboost_model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            use_best_model=True
        )
        
        # Evaluate
        y_pred_log = self.catboost_model.predict(X_val)
        y_pred = np.expm1(y_pred_log)  # Inverse log transformation
        y_val_actual = np.expm1(y_val)
        
        smape = self.calculate_smape(y_val_actual, y_pred)
        mae = mean_absolute_error(y_val_actual, y_pred)
        
        print(f"CatBoost SMAPE: {smape:.2f}%")
        print(f"CatBoost MAE: {mae:.2f}")
        
        return smape
    
    def predict(self, X):
        """Make predictions with CatBoost"""
        print("Making predictions...")
        
        # Predict with log transformation
        predictions_log = self.catboost_model.predict(X)
        
        # Inverse log transformation
        predictions = np.expm1(predictions_log)
        
        # Ensure positive predictions
        predictions = np.maximum(predictions, 0.01)
        
        return predictions

def main():
    """Fast CatBoost training and prediction pipeline"""
    print("=== Fast CatBoost ML Challenge 2025: Smart Product Pricing Solution ===")
    print("Target: 20-30% SMAPE using optimized CatBoost")
    
    start_time = time.time()
    
    # Initialize predictor
    predictor = FastCatBoostPricingPredictor()
    
    # Load data
    print("Loading data...")
    train_df = pd.read_csv('dataset/train.csv')
    test_df = pd.read_csv('dataset/test.csv')
    
    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    
    # Use larger sample for better performance
    train_sample_size = 60000  # Increased for better performance
    print(f"Using {train_sample_size} samples for training...")
    train_sample = train_df.sample(n=min(train_sample_size, len(train_df)), random_state=42)
    
    # Prepare features
    X_train = predictor.prepare_features(train_sample, is_training=True)
    y_train = train_sample['price'].values
    
    print(f"Feature matrix shape: {X_train.shape}")
    print(f"Number of features: {X_train.shape[1]}")
    
    # Train CatBoost model
    smape = predictor.train_catboost_model(X_train, y_train)
    
    # Prepare test features
    print("Preparing test features...")
    X_test = predictor.prepare_features(test_df, is_training=False)
    
    # Make predictions
    predictions = predictor.predict(X_test)
    
    # Create submission file
    submission_df = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': predictions
    })
    
    # Save predictions
    output_file = 'dataset/test_out_fast_catboost.csv'
    submission_df.to_csv(output_file, index=False)
    
    end_time = time.time()
    
    print(f"\nFast CatBoost predictions saved to {output_file}")
    print(f"Total predictions: {len(submission_df)}")
    print(f"Execution time: {end_time - start_time:.2f} seconds")
    
    # Display statistics
    print(f"\nFast CatBoost Prediction Statistics:")
    print(f"Mean price: ${predictions.mean():.2f}")
    print(f"Median price: ${np.median(predictions):.2f}")
    print(f"Min price: ${predictions.min():.2f}")
    print(f"Max price: ${predictions.max():.2f}")
    print(f"Std price: ${predictions.std():.2f}")
    
    print(f"\nModel Performance:")
    print(f"CatBoost SMAPE: {smape:.2f}%")
    print(f"Target SMAPE: 20-30%")
    print(f"Performance: {'EXCELLENT' if smape <= 30 else 'GOOD' if smape <= 40 else 'NEEDS IMPROVEMENT'}")
    
    return predictor, submission_df

if __name__ == "__main__":
    predictor, submission = main()
