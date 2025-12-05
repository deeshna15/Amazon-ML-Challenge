"""
Fast 40% SMAPE ML Challenge 2025: Smart Product Pricing Solution
Target: 40% SMAPE with ultra-fast execution
Optimized for speed while maintaining good accuracy
"""

import os
import re
import pandas as pd
import numpy as np
import time
from catboost import CatBoostRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class Fast40SMAPEPredictor:
    def __init__(self):
        self.catboost_model = None
        self.text_vectorizer = TfidfVectorizer(
            max_features=1000, 
            stop_words='english', 
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.9
        )
        self.scaler = RobustScaler()
        self.label_encoders = {}
        
    def extract_fast_features(self, catalog_content):
        """Extract optimized features for 40% SMAPE target"""
        features = {}
        
        # Essential text statistics
        features['text_length'] = len(catalog_content)
        features['word_count'] = len(catalog_content.split())
        features['sentence_count'] = len(re.findall(r'[.!?]+', catalog_content))
        features['unique_word_ratio'] = len(set(catalog_content.lower().split())) / max(len(catalog_content.split()), 1)
        features['avg_word_length'] = np.mean([len(word) for word in catalog_content.split()]) if catalog_content.split() else 0
        features['caps_ratio'] = sum(1 for c in catalog_content if c.isupper()) / max(len(catalog_content), 1)
        features['digit_ratio'] = sum(1 for c in catalog_content if c.isdigit()) / max(len(catalog_content), 1)
        
        # Critical structured data
        value_match = re.search(r'Value: ([\d.]+)', catalog_content)
        features['value'] = float(value_match.group(1)) if value_match else 0.0
        
        unit_match = re.search(r'Unit: ([^\n]+)', catalog_content)
        features['unit'] = unit_match.group(1).strip() if unit_match else 'unknown'
        
        # Bullet points analysis
        bullet_points = re.findall(r'Bullet Point \d+:', catalog_content)
        features['bullet_points_count'] = len(bullet_points)
        
        bullet_content = re.findall(r'Bullet Point \d+: ([^\n]+)', catalog_content)
        features['bullet_text_length'] = sum(len(bp) for bp in bullet_content)
        features['bullet_density'] = len(bullet_points) / max(features['word_count'], 1)
        
        # Package information
        pack_match = re.search(r'Pack of (\d+)', catalog_content)
        features['pack_size'] = int(pack_match.group(1)) if pack_match else 1
        
        case_match = re.search(r'Case of (\d+)', catalog_content)
        features['case_size'] = int(case_match.group(1)) if case_match else 1
        
        # Size extraction
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
        features['size_unit'] = 'unknown'
        
        for unit, pattern in size_patterns.items():
            match = re.search(pattern, catalog_content, re.IGNORECASE)
            if match:
                features['has_size_info'] = 1
                features['size_value'] = float(match.group(1))
                features['size_unit'] = unit
                break
        
        # Quality indicators
        premium_indicators = ['Premium', 'Deluxe', 'Professional', 'Luxury', 'Organic', 'Natural']
        features['has_premium_indicators'] = any(indicator in catalog_content for indicator in premium_indicators)
        
        brand_indicators = ['Brand:', 'by', 'Made by', 'Manufacturer:']
        features['has_brand_info'] = any(indicator in catalog_content for indicator in brand_indicators)
        
        package_indicators = ['Pack of', 'Bundle', 'Set of', 'Multi-pack', 'Case of', 'Bulk']
        features['is_packaged'] = any(indicator in catalog_content for indicator in package_indicators)
        
        safety_indicators = ['Organic', 'Natural', 'Non-GMO', 'Gluten Free', 'Vegan']
        features['has_safety_indicators'] = any(indicator in catalog_content for indicator in safety_indicators)
        
        price_indicators = ['Sale', 'Discount', 'Special', 'Limited', 'Clearance', 'Bulk']
        features['has_price_indicators'] = any(indicator in catalog_content for indicator in price_indicators)
        
        # Numeric analysis
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', catalog_content)
        if numbers:
            numeric_values = [float(n) for n in numbers]
            features['numeric_count'] = len(numeric_values)
            features['max_number'] = max(numeric_values)
            features['min_number'] = min(numeric_values)
            features['avg_number'] = np.mean(numeric_values)
            features['median_number'] = np.median(numeric_values)
            
            # Large numbers analysis
            large_numbers = [n for n in numeric_values if n > 10]
            features['large_number_count'] = len(large_numbers)
            features['large_number_avg'] = np.mean(large_numbers) if large_numbers else 0
            
            # Very large numbers (bulk items)
            very_large = [n for n in numeric_values if n > 100]
            features['very_large_count'] = len(very_large)
            features['very_large_avg'] = np.mean(very_large) if very_large else 0
            
            # Price-like numbers
            price_like = [n for n in numeric_values if 5 <= n <= 500]
            features['price_like_count'] = len(price_like)
            features['price_like_avg'] = np.mean(price_like) if price_like else 0
        else:
            features.update({
                'numeric_count': 0, 'max_number': 0, 'min_number': 0, 'avg_number': 0,
                'median_number': 0, 'large_number_count': 0, 'large_number_avg': 0,
                'very_large_count': 0, 'very_large_avg': 0,
                'price_like_count': 0, 'price_like_avg': 0
            })
        
        # Category detection
        category_keywords = {
            'food': ['food', 'snack', 'candy', 'chocolate', 'sauce', 'spice', 'beverage', 'drink', 'tea', 'coffee'],
            'electronics': ['electronic', 'digital', 'battery', 'cable', 'charger', 'device', 'phone', 'tablet'],
            'beauty': ['beauty', 'cosmetic', 'makeup', 'skincare', 'shampoo', 'lotion', 'cream'],
            'home': ['kitchen', 'cookware', 'furniture', 'decor', 'cleaning', 'storage'],
            'health': ['vitamin', 'supplement', 'medicine', 'health', 'fitness', 'medical', 'protein'],
            'clothing': ['shirt', 'pants', 'dress', 'jacket', 'shoes', 'clothing', 'apparel']
        }
        
        for category, keywords in category_keywords.items():
            features[f'is_{category}'] = any(keyword in catalog_content.lower() for keyword in keywords)
        
        # Essential ratios
        features['value_to_word_ratio'] = features['value'] / max(features['word_count'], 1)
        features['value_to_text_ratio'] = features['value'] / max(features['text_length'], 1)
        features['pack_size_to_value_ratio'] = features['pack_size'] / max(features['value'], 1)
        features['size_value_to_value_ratio'] = features['size_value'] / max(features['value'], 1)
        features['bullet_to_word_ratio'] = features['bullet_points_count'] / max(features['word_count'], 1)
        
        return features
    
    def calculate_smape(self, y_true, y_pred):
        """Calculate SMAPE metric"""
        return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)))
    
    def prepare_fast_features(self, df, is_training=True, sample_size=None):
        """Prepare fast features for 40% SMAPE target"""
        if sample_size and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
        
        # Extract features
        features_list = []
        for idx, catalog_content in enumerate(df['catalog_content']):
            if idx % 3000 == 0 and idx > 0:
                print(f"Processing features: {idx}/{len(df)}")
            features_list.append(self.extract_fast_features(catalog_content))
        
        feature_df = pd.DataFrame(features_list)
        feature_df = feature_df.fillna(0)
        
        # Categorical encoding
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
        
        # TF-IDF features
        if is_training:
            text_features = self.text_vectorizer.fit_transform(df['catalog_content'])
            self.text_feature_names = [f'text_{i}' for i in range(text_features.shape[1])]
        else:
            text_features = self.text_vectorizer.transform(df['catalog_content'])
        
        text_df = pd.DataFrame(text_features.toarray(), columns=self.text_feature_names)
        
        # Combine features
        final_features = pd.concat([feature_df, text_df], axis=1)
        final_features = final_features.drop(columns=['unit', 'size_unit'], errors='ignore')
        
        # Convert all features to numeric
        for col in final_features.columns:
            final_features[col] = pd.to_numeric(final_features[col], errors='coerce').fillna(0)
        
        return final_features
    
    def train_fast_catboost(self, X, y):
        """Train fast CatBoost for 40% SMAPE target"""
        print("Training fast CatBoost model...")
        
        # Moderate outlier removal
        price_98th = np.percentile(y, 98)
        price_2nd = np.percentile(y, 2)
        mask = (y >= price_2nd) & (y <= price_98th)
        X_clean = X[mask]
        y_clean = y[mask]
        
        print(f"Training on {len(X_clean)} samples after outlier removal")
        print(f"Feature matrix shape: {X_clean.shape}")
        
        # Robust scaling
        X_scaled = self.scaler.fit_transform(X_clean)
        
        # Log transformation
        y_log = np.log1p(y_clean)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_log, test_size=0.2, random_state=42)
        
        # Optimized CatBoost parameters for 40% SMAPE
        self.catboost_model = CatBoostRegressor(
            iterations=1200,
            learning_rate=0.05,
            depth=9,
            l2_leaf_reg=3,
            loss_function='RMSE',
            random_seed=42,
            verbose=False,
            early_stopping_rounds=50,
            bootstrap_type='Bayesian',
            bagging_temperature=1,
            random_strength=1,
            rsm=0.8,
            min_data_in_leaf=3,
            max_leaves=31,
            grow_policy='Lossguide',
            feature_border_type='GreedyLogSum',
            boosting_type='Plain',
            leaf_estimation_method='Newton',
            score_function='Cosine'
        )
        
        self.catboost_model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            use_best_model=True
        )
        
        # Evaluation
        y_pred_log = self.catboost_model.predict(X_val)
        y_pred = np.expm1(y_pred_log)
        y_val_actual = np.expm1(y_val)
        
        smape = self.calculate_smape(y_val_actual, y_pred)
        mae = mean_absolute_error(y_val_actual, y_pred)
        
        print(f"Fast CatBoost SMAPE: {smape:.2f}%")
        print(f"Fast CatBoost MAE: {mae:.2f}")
        
        return smape
    
    def predict_fast(self, X):
        """Make fast predictions"""
        X_scaled = self.scaler.transform(X)
        predictions_log = self.catboost_model.predict(X_scaled)
        predictions = np.expm1(predictions_log)
        predictions = np.maximum(predictions, 0.01)
        return predictions

def main():
    """Fast 40% SMAPE training and prediction pipeline"""
    print("=== Fast 40% SMAPE ML Challenge 2025: Smart Product Pricing Solution ===")
    print("Target: 40% SMAPE with ultra-fast execution")
    
    start_time = time.time()
    
    # Initialize predictor
    predictor = Fast40SMAPEPredictor()
    
    # Load data
    print("Loading data...")
    train_df = pd.read_csv('dataset/train.csv')
    test_df = pd.read_csv('dataset/test.csv')
    
    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    
    # Optimized sample size for 40% SMAPE target
    train_sample_size = 15000  # Balanced sample for speed and accuracy
    print(f"Using {train_sample_size} samples for training...")
    train_sample = train_df.sample(n=min(train_sample_size, len(train_df)), random_state=42)
    
    # Prepare features
    print("Preparing fast features...")
    X_train = predictor.prepare_fast_features(train_sample, is_training=True)
    y_train = train_sample['price'].values
    
    print(f"Feature matrix shape: {X_train.shape}")
    
    # Train model
    smape = predictor.train_fast_catboost(X_train, y_train)
    
    # Fast test processing
    print("Making fast predictions...")
    chunk_size = 20000  # Larger chunks for speed
    all_predictions = []
    all_sample_ids = []
    
    for i in range(0, len(test_df), chunk_size):
        chunk = test_df.iloc[i:i+chunk_size]
        print(f"Processing chunk {i//chunk_size + 1}/{(len(test_df)-1)//chunk_size + 1}")
        
        X_chunk = predictor.prepare_fast_features(chunk, is_training=False)
        chunk_predictions = predictor.predict_fast(X_chunk)
        
        all_predictions.extend(chunk_predictions)
        all_sample_ids.extend(chunk['sample_id'].tolist())
    
    # Create submission file
    submission_df = pd.DataFrame({
        'sample_id': all_sample_ids,
        'price': all_predictions
    })
    
    # Save predictions
    output_file = 'dataset/test_out_fast_40_smape.csv'
    submission_df.to_csv(output_file, index=False)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    print(f"\nFast predictions saved to {output_file}")
    print(f"Total predictions: {len(submission_df)}")
    print(f"Execution time: {execution_time:.2f} seconds ({execution_time/60:.1f} minutes)")
    
    # Display statistics
    print(f"\nPrediction Statistics:")
    print(f"Mean price: ${submission_df['price'].mean():.2f}")
    print(f"Median price: ${submission_df['price'].median():.2f}")
    print(f"Min price: ${submission_df['price'].min():.2f}")
    print(f"Max price: ${submission_df['price'].max():.2f}")
    print(f"Std price: ${submission_df['price'].std():.2f}")
    
    print(f"\nPerformance Summary:")
    print(f"Fast CatBoost SMAPE: {smape:.2f}%")
    print(f"Target SMAPE: 40%")
    print(f"Speed: {'ULTRA FAST' if execution_time < 300 else 'FAST' if execution_time < 600 else 'MODERATE'}")
    print(f"Accuracy: {'EXCELLENT' if smape <= 40 else 'GOOD' if smape <= 45 else 'NEEDS IMPROVEMENT'}")
    
    # Target assessment
    if smape <= 40:
        print("\n[SUCCESS] Target 40% SMAPE achieved!")
    elif smape <= 45:
        print("\n[SUCCESS] Close to target range!")
    else:
        print("\n[PROGRESS] Solution completed, room for optimization")
    
    return predictor, submission_df

if __name__ == "__main__":
    predictor, submission = main()
