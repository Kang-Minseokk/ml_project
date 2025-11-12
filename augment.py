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
    
    
    def main():                                        
    
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
