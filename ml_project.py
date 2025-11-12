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
# 1. DATA LOADING AND AUGMENTATION
# ============================================================================

class DataLoader:
    """Load and manage dataset from directory structure"""
    
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.data = []
        self.labels = []
        self.categories = []
        
    def load_data(self):
        """Load all txt files from category folders"""
        print("=" * 70)
        print("STEP 1: DATA LOADING AND EXPLORATION")
        print("=" * 70)
        
        data_path = self.base_path / "1st_data"
        
        for category_dir in sorted(data_path.iterdir()):
            if not category_dir.is_dir():
                continue
                
            category_name = category_dir.name
            self.categories.append(category_name)
            print(f"\nLoading category: {category_name}")
            
            txt_files = sorted(category_dir.glob("*.txt"))
            print(f"  Found {len(txt_files)} files")
            
            for txt_file in txt_files:
                features = self._parse_txt_file(txt_file)
                if features is not None:
                    self.data.append(features)
                    self.labels.append(category_name)
        
        print(f"\nTotal samples loaded: {len(self.data)}")
        print(f"Categories: {self.categories}")
        print(f"Label distribution:")
        for cat in self.categories:
            count = self.labels.count(cat)
            print(f"  {cat}: {count} samples")
        
        # Pad or truncate features to same length
        if len(self.data) > 0:
            # Find max length
            max_len = max(len(f) for f in self.data)
            print(f"\nFeature vector length: {max_len}")
            
            # Pad all features to max length
            padded_data = []
            for features in self.data:
                if len(features) < max_len:
                    padded = np.pad(features, (0, max_len - len(features)), mode='constant')
                else:
                    padded = features[:max_len]
                padded_data.append(padded)
            
            return np.array(padded_data), np.array(self.labels)
        
        return np.array(self.data), np.array(self.labels)
    
    def _parse_txt_file(self, filepath):
        """
        Parse a single txt file
        Format: r,<id>,<timestamp>,<values_group1>,<values_group2>,...
        Extract numeric features from the file
        """
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            # Collect all numeric values from all lines
            all_values = []
            for line in lines:
                if line.startswith('r,') or line.startswith('s,'):
                    # Parse comma-separated values
                    parts = line.strip().split(',')
                    # Extract numeric values (skip first 3 parts: type, id, timestamp)
                    for part in parts[3:]:
                        if part and part != '#':
                            try:
                                # Handle values with slashes (e.g., "61/63/38/61")
                                if '/' in part:
                                    numeric_parts = [int(p) for p in part.split('/')]
                                    all_values.extend(numeric_parts)
                                else:
                                    all_values.append(int(part))
                            except (ValueError, TypeError):
                                pass
            
            if len(all_values) == 0:
                return None
            
            # Convert to numpy array and return as feature vector
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
    def generate_synthetic_samples(X, y, noise_factor=0.05, n_synthetic_per_sample=2):
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
            max_depth=15,
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
# 3. CUSTOM EVALUATION METRICS
# ============================================================================

class CustomEvaluator:
    """
    Custom evaluation framework with multiple metrics
    Metrics designed for imbalanced multi-class classification
    """
    
    def __init__(self, model, X_test, y_test, classes):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.classes = classes
        self.y_pred = model.predict(X_test)
        self.y_pred_proba = model.predict_proba(X_test)
        
    def evaluate(self):
        """Compute all evaluation metrics"""
        print("\n" + "=" * 70)
        print("STEP 3: MODEL EVALUATION WITH CUSTOM METRICS")
        print("=" * 70)
        
        results = {}
        
        # 1. Overall Accuracy
        accuracy = accuracy_score(self.y_test, self.y_pred)
        results['overall_accuracy'] = accuracy
        print(f"\n[METRIC 1] Overall Accuracy: {accuracy:.4f}")
        
        # 2. Per-class Metrics (Precision, Recall, F1)
        print("\n[METRIC 2] Per-Class Metrics (Precision, Recall, F1):")
        precision = precision_score(self.y_test, self.y_pred, average=None)
        recall = recall_score(self.y_test, self.y_pred, average=None)
        f1 = f1_score(self.y_test, self.y_pred, average=None)
        
        for i, cls in enumerate(self.classes):
            print(f"  {cls:20s}: P={precision[i]:.4f}, R={recall[i]:.4f}, F1={f1[i]:.4f}")
        
        results['precision_per_class'] = dict(zip(self.classes, precision))
        results['recall_per_class'] = dict(zip(self.classes, recall))
        results['f1_per_class'] = dict(zip(self.classes, f1))
        
        # 3. Macro and Weighted Averages (for imbalanced data)
        print("\n[METRIC 3] Averaged Metrics (for imbalanced data):")
        macro_f1 = f1_score(self.y_test, self.y_pred, average='macro')
        weighted_f1 = f1_score(self.y_test, self.y_pred, average='weighted')
        print(f"  Macro F1 (unweighted): {macro_f1:.4f}")
        print(f"  Weighted F1 (class-size weighted): {weighted_f1:.4f}")
        
        results['macro_f1'] = macro_f1
        results['weighted_f1'] = weighted_f1
        
        # 4. Confusion Matrix Analysis
        print("\n[METRIC 4] Confusion Matrix Analysis:")
        cm = confusion_matrix(self.y_test, self.y_pred)
        
        # Per-class error rates
        print("  Per-class Error Rates:")
        for i, cls in enumerate(self.classes):
            total = cm[i].sum()
            correct = cm[i, i]
            error_rate = 1 - (correct / total) if total > 0 else 0
            print(f"    {cls}: {error_rate:.4f} ({total - correct}/{total} errors)")
        
        results['confusion_matrix'] = cm.tolist()
        
        # 5. Balanced Accuracy (handles class imbalance)
        print("\n[METRIC 5] Balanced Accuracy (per-class average recall):")
        balanced_acc = recall.mean()
        print(f"  Balanced Accuracy: {balanced_acc:.4f}")
        results['balanced_accuracy'] = balanced_acc
        
        # 6. Confidence Analysis
        print("\n[METRIC 6] Prediction Confidence Analysis:")
        max_proba = np.max(self.y_pred_proba, axis=1)
        avg_confidence = np.mean(max_proba)
        std_confidence = np.std(max_proba)
        print(f"  Average Confidence: {avg_confidence:.4f}")
        print(f"  Confidence Std Dev: {std_confidence:.4f}")
        print(f"  Confidence Range: [{np.min(max_proba):.4f}, {np.max(max_proba):.4f}]")
        
        results['avg_confidence'] = avg_confidence
        results['confidence_std'] = std_confidence
        
        # 7. ROC-AUC for binary comparisons (One-vs-Rest)
        print("\n[METRIC 7] ROC-AUC Scores (One-vs-Rest for each class):")
        try:
            # For multi-class, compute OvR AUC
            from sklearn.preprocessing import label_binarize
            y_test_bin = label_binarize(self.y_test, classes=self.classes)
            
            for i, cls in enumerate(self.classes):
                if len(np.unique(self.y_test)) > 1:
                    try:
                        auc_score = roc_auc_score(y_test_bin[:, i], self.y_pred_proba[:, i])
                        print(f"  {cls}: {auc_score:.4f}")
                        results[f'roc_auc_{cls}'] = auc_score
                    except:
                        print(f"  {cls}: N/A (single class in test set)")
        except Exception as e:
            print(f"  AUC calculation skipped: {e}")
        
        # 8. Error Distribution Analysis
        print("\n[METRIC 8] Error Type Analysis:")
        errors = self.y_test != self.y_pred
        error_rate = errors.mean()
        print(f"  Total Error Rate: {error_rate:.4f}")
        
        if error_rate > 0:
            print(f"  Most Common Misclassification:")
            for i, pred_class in enumerate(self.y_pred[errors]):
                true_class = self.y_test[errors][i]
                print(f"    {true_class} -> {pred_class}")
        
        results['total_error_rate'] = error_rate
        
        return results
    
    def visualize_results(self, save_path="results"):
        """Generate visualizations"""
        os.makedirs(save_path, exist_ok=True)
        
        # 1. Confusion Matrix Heatmap
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(self.y_test, self.y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.classes, yticklabels=self.classes)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'confusion_matrix.png'), dpi=300)
        plt.close()
        
        # 2. Per-class F1 Scores
        f1 = f1_score(self.y_test, self.y_pred, average=None)
        plt.figure(figsize=(10, 6))
        plt.bar(self.classes, f1, color='steelblue')
        plt.ylabel('F1 Score')
        plt.xlabel('Class')
        plt.title('Per-Class F1 Scores')
        plt.ylim([0, 1.0])
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'f1_scores.png'), dpi=300)
        plt.close()
        
        # 3. Confidence Distribution
        max_proba = np.max(self.y_pred_proba, axis=1)
        plt.figure(figsize=(10, 6))
        plt.hist(max_proba, bins=30, color='skyblue', edgecolor='black')
        plt.xlabel('Prediction Confidence')
        plt.ylabel('Frequency')
        plt.title('Distribution of Prediction Confidence')
        plt.axvline(np.mean(max_proba), color='red', linestyle='--', label=f'Mean: {np.mean(max_proba):.3f}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'confidence_distribution.png'), dpi=300)
        plt.close()
        
        # 4. Accuracy by Class
        accuracy_per_class = []
        for i, cls in enumerate(self.classes):
            mask = self.y_test == cls
            if mask.sum() > 0:
                acc = accuracy_score(self.y_test[mask], self.y_pred[mask])
                accuracy_per_class.append(acc)
            else:
                accuracy_per_class.append(0)
        
        plt.figure(figsize=(10, 6))
        plt.bar(self.classes, accuracy_per_class, color='lightgreen')
        plt.ylabel('Accuracy')
        plt.xlabel('Class')
        plt.title('Per-Class Accuracy')
        plt.ylim([0, 1.0])
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'accuracy_by_class.png'), dpi=300)
        plt.close()
        
        print(f"\nVisualizations saved to '{save_path}/' directory")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Execute the full ML pipeline"""
    
    base_path = r"c:\Projects\2025_ml_project"
    
    # Step 1: Load and augment data
    print("\n" + "=" * 70)
    print("MACHINE LEARNING PROJECT: SHAPE CLASSIFICATION")
    print("=" * 70)
    
    loader = DataLoader(base_path)
    X, y = loader.load_data()
    
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
    model = ShapeClassifierModel(n_estimators=150)
    model.train(X_train, y_train)
    
    # Step 3: Evaluate model
    evaluator = CustomEvaluator(model, X_test, y_test, model.classes_)
    results = evaluator.evaluate()
    
    # Visualize results
    evaluator.visualize_results(save_path=os.path.join(base_path, 'results'))
    
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
