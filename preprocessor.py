"""
Machine Learning Project: Shape/Motion Pattern Classification
Sub-quests:
1. Augment or refine the dataset by yourself
2. Build a model that classify the input into the given categories
3. Suggesting an evaluator of your own model with self-set error metrics and proving its validity
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score, 
    roc_curve, auc
)
import seaborn as sns
from datetime import datetime
import json

# ============================================================================
# 1. DataPreprocessor의 두 가지 역할 : data_path에 있는 데이터를 save_root에 저장을 하는 역할과 
#    우리가 필요로 하는 데이터인 x, y, z라고 하는 좌표만으로 이루어진 구조로 데이터를 전처리하는 역할
# ============================================================================

class DataPreprocessor:        
    def __init__(self):        
        self.data = []
        self.labels = []
        self.categories = []
        
    def load_data(self):        
        print("=" * 70)
        print("STEP 1: 데이터 로딩을 하고, 전처리 과정을 거쳐봅시다")
        print("=" * 70)
        
        data_path =  Path("raw_data")
        save_root = Path("preprocessed_data")
        
        for category_dir in sorted(data_path.iterdir()):
            if not category_dir.is_dir():
                continue
                
            category_name = category_dir.name                        
            self.categories.append(category_name)
            print(f"\nLoading category: {category_name}")
            
            # 저장을 하기 위한 디렉토리를 생성합니다
            (save_root / category_name).mkdir(parents=True, exist_ok=True)
            
            txt_files = sorted(category_dir.glob("*.txt"))
            print(f"  Found {len(txt_files)} files")
            
            for txt_file in txt_files:
                features = self._parse_txt_file(txt_file)                
                if features is not None:
                    self.data.append(features)
                    self.labels.append(category_name)
                    
                    # 실제로 저장하게 되는 데이터를 가공해봅니다.
                    arr = features.reshape(-1, 3)
                    
                    save_path = save_root / category_name / txt_file.name
                    
                    with open(save_path, "w") as f:
                        for row in arr:
                            f.write(f"{int(row[0])},{int(row[1])},{int(row[2])}\n")
                    
                    print(f"    전처리 데이터 저장 완료! -> {save_path}")
            breakpoint()        
                                    
    def _parse_txt_file(self, filepath):
        """
        Parse a single txt file
        Format: r,<id>,<timestamp>,<values_group1>,...
        Extract only index=6 (7th item) as numeric feature
        """
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()                
            
            all_values = []

            for line in lines:
                if line.startswith('r,') or line.startswith('s,'):
                    parts = line.strip().split(',')

                    # Ensure index 6 exists
                    if len(parts) > 6:
                        value = parts[6]   # 7th column (index 6)                        

                        if value and value != '#':
                            try:
                                # e.g. "61/63/38/61"
                                if '/' in value:
                                    nums = [int(p) for p in value.split('/')]
                                    all_values.extend(nums)
                                else:
                                    all_values.append(int(value))
                            except (ValueError, TypeError):
                                pass
            
            # breakpoint()
            
            if len(all_values) == 0:
                return None

            return np.array(all_values, dtype=np.float32)
        
        except Exception as e:
            print(f"  Error reading {filepath}: {e}")
            return None


class DataAugmentation:
    """
    Augment dataset through:
    1. Feature engineering (statistical aggregations)
    2. Synthetic sample generation (small perturbations)
    3. Temporal windowing
    """
    
    @staticmethod
    def augment_features(X, window_sizes=[3, 5]):
        """
        Create augmented features from raw feature vector
        - Original features
        - Statistical features (mean, std, min, max of windows)
        - Differences/derivatives
        """
        print("\nAugmenting features...")
        augmented = [X]
        
        for window_size in window_sizes:
            if len(X) > window_size:
                # Rolling mean features
                windowed = np.convolve(X, np.ones(window_size)/window_size, mode='valid')
                augmented.append(windowed[:len(X)])
                
                # Rolling std features
                windowed_std = np.array([
                    np.std(X[max(0, i-window_size):i+1]) 
                    for i in range(len(X))
                ])
                augmented.append(windowed_std)
        
        # Add derivative (differences)
        derivatives = np.diff(X, prepend=X[0])
        augmented.append(derivatives)
        
        # Combine and pad to same length
        min_len = min(len(a) for a in augmented)
        augmented = np.array([a[:min_len] for a in augmented])
        
        return augmented.flatten()
    
    @staticmethod
    def generate_synthetic_samples(X, y, noise_factor=0.1, n_synthetic_per_sample=2):
        """
        Generate synthetic samples by adding small Gaussian noise
        This helps with data augmentation and prevents overfitting
        """
        print(f"\nGenerating synthetic samples (noise_factor={noise_factor})...")
        X_synthetic = list(X)  # Start with original samples
        y_synthetic = list(y)
        
        for i, sample in enumerate(X):
            for _ in range(n_synthetic_per_sample):
                # Add Gaussian noise scaled by standard deviation
                noise = np.random.normal(0, noise_factor * np.std(sample), size=sample.shape)
                synthetic_sample = sample + noise
                X_synthetic.append(synthetic_sample)
                y_synthetic.append(y[i])
        
        print(f"Original: {len(X)} samples -> Augmented: {len(X_synthetic)} samples")
        return np.array(X_synthetic), np.array(y_synthetic)


# ============================================================================
# 2. MODEL BUILDING
# ============================================================================

class ShapeClassifierModel:
    """
    Multi-class classifier for shape/motion patterns
    Uses Random Forest with augmented features
    """
    
    def __init__(self, n_estimators=100, random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=2,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
            class_weight='balanced'
        )
        self.scaler = StandardScaler()
        self.feature_importance = None
        self.classes_ = None
        
    def train(self, X_train, y_train):
        """Train the model"""
        print("\n" + "=" * 70)
        print("STEP 2: MODEL TRAINING")
        print("=" * 70)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train model
        print(f"\nTraining Random Forest with {len(X_train)} samples...")
        self.model.fit(X_train_scaled, y_train)
        self.classes_ = self.model.classes_
        
        # Get feature importance
        self.feature_importance = self.model.feature_importances_
        
        # Training accuracy
        train_pred = self.model.predict(X_train_scaled)
        train_acc = accuracy_score(y_train, train_pred)
        print(f"Training Accuracy: {train_acc:.4f}")
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():          
    # x, y, z좌표만을 추출해내는 작업도 함께 DataPreprocessor가 해줍니다                  
    loader = DataPreprocessor()
    X, y = loader.load_data() # 여기는 문제가 없어
    
    for i in range(len(X)):
        record = X[i]

        # (1899,) → (-1, 3)
        try:
            coords = record.reshape(-1, 3)
        except:
            print(f"[!] Row {i} shape {record.shape} cannot reshape into (-1,3)")
            continue

        # (0,0,0) 제거
        coords = coords[~(coords == 0).all(axis=1)]

        # 저장 경로
        filepath = os.path.join("coords_txt", f"{i}.txt")

        # 항상 새로 만들기 (없으면 생성, 있으면 overwrite)
        with open(filepath, "w") as f:
            for (x, y, z) in coords:
                f.write(f"{x} {y} {z}\n")

        print(f"Saved: {filepath}")
    
    breakpoint()    
    
    # Feature engineering and augmentation
    print("\n" + "=" * 70)
    print("AUGMENTING DATASET")
    print("=" * 70)
    
    X_augmented = np.array([DataAugmentation.augment_features(sample) for sample in X])
    print(f"Feature shape after augmentation: {X_augmented.shape}")
    
    # Optional: Generate synthetic samples
    use_synthetic = True
    if use_synthetic:
        X_augmented, y = DataAugmentation.generate_synthetic_samples(
            X_augmented, y, noise_factor=0.08, n_synthetic_per_sample=1
        )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_augmented, y, test_size=0.2, random_state=42, stratify=y
    )    
    
    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Step 2: Train model
    model = ShapeClassifierModel(n_estimators=2)
    model.train(X_train, y_train)
    
    # Step 3: Evaluate model
    evaluator = CustomEvaluator(model, X_test, y_test, model.classes_)
    results = evaluator.evaluate()
    
    # Visualize results
    evaluator.visualize_results(save_path=os.path.join('visualization', 'results'))
    
    # Save results
    results_json = {k: v for k, v in results.items() if k != 'confusion_matrix'}
    with open(os.path.join(base_path, 'results', 'evaluation_metrics.json'), 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print("\n" + "=" * 70)
    print("PROJECT COMPLETED")
    print("=" * 70)
    print(f"Results saved to: {os.path.join(base_path, 'results')}")
    
    return model, results


if __name__ == "__main__":
    model, results = main()

