import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import data_load
from ml_project import DataLoader, DataAugmentation


def plot_sample(idx: int = 0, kind: str = 'line', save_path: str | None = None, show: bool = False):
	"""Plot a sample from data_load.

	- idx: index into data_load.data / data_load.labels
	- kind: 'line' or 'scatter'
	- save_path: if provided, the plot will be saved to this path (PNG supported)
	- show: if True, call plt.show()
	"""
	# Ensure data is loaded
	if not data_load.data:
		data_load.process()

	if not data_load.data:
		raise SystemExit("No data available after running data_load.process(). Check base_dir and files.")

	if idx < 0 or idx >= len(data_load.data):
		raise IndexError(f"Index out of range: {idx} (available: 0..{len(data_load.data)-1})")

	sample = data_load.data[idx]
	label = data_load.labels[idx]

	fig = plt.figure(figsize=(8, 6))
	ax = fig.add_subplot(111, projection='3d')

	if kind == 'line':
		ax.plot(sample['x'], sample['y'], sample['z'])
	else:
		ax.scatter(sample['x'], sample['y'], sample['z'], s=8)

	ax.set_title(label)
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('z')

	if save_path:
		# ensure directory exists
		os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
		fig.savefig(save_path, dpi=200)
		print(f"Saved plot to {save_path}")

	if show:
		plt.show()

	plt.close(fig)


def plot_all(save_dir: str = 'plots', kind: str = 'line', show: bool = False):
	"""Generate and save plots for all samples in data_load.data.

	Files are saved to: <save_dir>/<label>/<index>_<label>.png
	Returns number of files saved.
	"""
	# Ensure data is loaded
	if not data_load.data:
		data_load.process()

	if not data_load.data:
		raise SystemExit("No data available to plot. Check data_load.base_dir and files.")

	saved = 0
	for i, sample in enumerate(data_load.data):
		label = data_load.labels[i]
		# safe folder name
		label_dir = os.path.join(save_dir, str(label).replace(' ', '_'))
		os.makedirs(label_dir, exist_ok=True)
		filename = f"{i:03d}_{label}.png"
		out_path = os.path.join(label_dir, filename)
		try:
			plot_sample(idx=i, kind=kind, save_path=out_path, show=show)
			saved += 1
		except Exception as e:
			print(f"Failed to plot sample {i} ({label}): {e}")

	print(f"Generated {saved} plots under '{save_dir}'")
	return saved


def load_augmented_data(base_path=None, noise_factor=0.05, n_synthetic=2):
	"""Load data using existing data_load and create augmented/synthetic samples.
	
	Returns:
		X_original: original feature vectors
		X_augmented: feature-engineered vectors  
		X_synthetic: synthetic samples with noise
		y_original, y_synthetic: corresponding labels
	"""
	print(f"Loading augmented data (noise_factor={noise_factor})...")
	
	# Use existing data_load module since ml_project expects different structure
	if not data_load.data:
		data_load.process()
	
	if not data_load.data:
		raise RuntimeError("No data available from data_load.process()")
	
	print(f"Converting {len(data_load.data)} samples to feature vectors...")
	
	# Convert data_load format to feature vectors
	X_original = []
	y_original = []
	for i, sample in enumerate(data_load.data):
		# Flatten x,y,z coordinates into feature vector
		coords = np.column_stack([sample['x'], sample['y'], sample['z']])
		features = coords.flatten()
		X_original.append(features)
		y_original.append(data_load.labels[i])
	
	# Pad to same length (find max length first)
	if X_original:
		max_len = max(len(x) for x in X_original)
		print(f"Padding feature vectors to length {max_len}")
		
		X_padded = []
		for features in X_original:
			if len(features) < max_len:
				padded = np.pad(features, (0, max_len - len(features)), mode='constant', constant_values=0)
			else:
				padded = features[:max_len]  # truncate if longer
			X_padded.append(padded)
		
		X_original = np.array(X_padded)
		y_original = np.array(y_original)
	
	print(f"Original data shape: {X_original.shape}")
	
	# Apply feature augmentation
	print("Applying feature augmentation...")
	X_augmented = np.array([DataAugmentation.augment_features(sample) for sample in X_original])
	print(f"Augmented data shape: {X_augmented.shape}")
	
	# Generate synthetic samples with noise
	print(f"Generating synthetic samples (noise_factor={noise_factor})...")
	X_synthetic, y_synthetic = DataAugmentation.generate_synthetic_samples(
		X_augmented, y_original, noise_factor=noise_factor, n_synthetic_per_sample=n_synthetic
	)
	print(f"Synthetic data shape: {X_synthetic.shape}")
	
	return X_original, X_augmented, X_synthetic, y_original, y_synthetic


def plot_original_vs_synthetic(original_idx=0, noise_factor=0.05, save_path=None, show=False):
	"""Plot comparison between original sample and its synthetic (noisy) versions.
	
	Args:
		original_idx: index of original sample to use as base
		noise_factor: noise level for synthetic generation
		save_path: path to save plot
		show: whether to display plot
	"""
	# Load augmented data
	X_orig, X_aug, X_synth, y_orig, y_synth = load_augmented_data(noise_factor=noise_factor)
	
	if original_idx >= len(X_orig):
		raise IndexError(f"Index {original_idx} out of range (max: {len(X_orig)-1})")
	
	# Find synthetic samples corresponding to this original
	original_label = y_orig[original_idx]
	
	# Get original sample
	orig_sample = X_orig[original_idx]
	
	# Find synthetic samples with same label (generated from this original)
	synthetic_indices = []
	count = 0
	for i, label in enumerate(y_synth):
		if label == original_label and i >= len(X_orig):  # synthetic samples come after originals
			synthetic_indices.append(i)
			count += 1
			if count >= 3:  # limit to 3 synthetic examples
				break
	
	# Create subplot comparison
	fig = plt.figure(figsize=(15, 10))
	
	# Plot original (as line plot of feature values)
	ax1 = plt.subplot(2, 2, 1)
	plt.plot(orig_sample, 'b-', linewidth=2, label='Original')
	plt.title(f'Original Sample (Label: {original_label})')
	plt.xlabel('Feature Index')
	plt.ylabel('Feature Value')
	plt.grid(True, alpha=0.3)
	plt.legend()
	
	# Plot augmented version
	ax2 = plt.subplot(2, 2, 2)
	aug_sample = X_aug[original_idx]
	plt.plot(aug_sample, 'g-', linewidth=2, label='Augmented')
	plt.title(f'Feature-Augmented Sample')
	plt.xlabel('Feature Index')
	plt.ylabel('Feature Value')
	plt.grid(True, alpha=0.3)
	plt.legend()
	
	# Plot synthetic versions
	ax3 = plt.subplot(2, 1, 2)
	plt.plot(orig_sample, 'b-', linewidth=3, label='Original', alpha=0.7)
	
	colors = ['red', 'orange', 'purple']
	for i, synth_idx in enumerate(synthetic_indices[:3]):
		synth_sample = X_synth[synth_idx][:len(orig_sample)]  # match length
		plt.plot(synth_sample, color=colors[i], linestyle='--', alpha=0.8, 
				label=f'Synthetic {i+1}')
	
	plt.title(f'Original vs Synthetic Samples (noise_factor={noise_factor})')
	plt.xlabel('Feature Index')
	plt.ylabel('Feature Value')
	plt.grid(True, alpha=0.3)
	plt.legend()
	
	plt.tight_layout()
	
	if save_path:
		os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
		fig.savefig(save_path, dpi=200, bbox_inches='tight')
		print(f"Saved comparison plot to {save_path}")
	
	if show:
		plt.show()
	else:
		plt.close(fig)


def plot_noise_comparison_grid(noise_levels=[0.02, 0.05, 0.1, 0.2], sample_idx=0, save_path=None, show=False):
	"""Plot grid showing effect of different noise levels on same original sample."""
	
	fig, axes = plt.subplots(2, 2, figsize=(15, 10))
	axes = axes.flatten()
	
	for i, noise_factor in enumerate(noise_levels):
		# Load data with this noise level
		X_orig, _, X_synth, y_orig, y_synth = load_augmented_data(noise_factor=noise_factor, n_synthetic=1)
		
		if sample_idx >= len(X_orig):
			continue
			
		original_sample = X_orig[sample_idx]
		original_label = y_orig[sample_idx]
		
		# Find first synthetic sample for this original
		synth_idx = len(X_orig) + sample_idx  # synthetic samples follow originals
		if synth_idx < len(X_synth):
			synthetic_sample = X_synth[synth_idx][:len(original_sample)]
			
			# Plot comparison
			ax = axes[i]
			ax.plot(original_sample, 'b-', linewidth=2, label='Original', alpha=0.8)
			ax.plot(synthetic_sample, 'r--', linewidth=2, label=f'Noise={noise_factor}', alpha=0.8)
			ax.set_title(f'Noise Factor: {noise_factor}')
			ax.set_xlabel('Feature Index')
			ax.set_ylabel('Feature Value')
			ax.grid(True, alpha=0.3)
			ax.legend()
	
	plt.suptitle(f'Effect of Different Noise Levels (Sample: {sample_idx}, Label: {y_orig[sample_idx]})')
	plt.tight_layout()
	
	if save_path:
		os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
		fig.savefig(save_path, dpi=200, bbox_inches='tight')
		print(f"Saved noise comparison grid to {save_path}")
	
	if show:
		plt.show()
	else:
		plt.close(fig)


def plot_all_synthetic(noise_factor=0.05, save_dir='synthetic_plots', n_synthetic=1):
	"""Generate plots for all original samples and their synthetic versions."""
	
	X_orig, X_aug, X_synth, y_orig, y_synth = load_augmented_data(noise_factor=noise_factor, n_synthetic=n_synthetic)
	
	os.makedirs(save_dir, exist_ok=True)
	
	saved = 0
	for i in range(len(X_orig)):
		try:
			# Create comparison plot for this sample
			label = y_orig[i]
			label_dir = os.path.join(save_dir, str(label).replace(' ', '_'))
			os.makedirs(label_dir, exist_ok=True)
			
			filename = f"{i:03d}_{label}_noise_{noise_factor}.png"
			out_path = os.path.join(label_dir, filename)
			
			plot_original_vs_synthetic(original_idx=i, noise_factor=noise_factor, 
									  save_path=out_path, show=False)
			saved += 1
			
		except Exception as e:
			print(f"Failed to plot sample {i}: {e}")
	
	print(f"Generated {saved} synthetic comparison plots under '{save_dir}'")
	return saved


def plot_synthetic_3d(noise_factor=0.05, save_dir='synthetic_3d_plots', kind='line'):
	"""Generate 3D trajectory plots for synthetic (noise-augmented) samples.
	
	This creates individual 3D plots for each synthetic sample, reconstructed from 
	the feature vectors by taking the original coordinates and adding noise.
	"""
	print(f"Creating 3D plots for synthetic data (noise_factor={noise_factor})...")
	
	# Ensure original data is loaded
	if not data_load.data:
		data_load.process()
	
	if not data_load.data:
		raise RuntimeError("No data available from data_load.process()")
	
	os.makedirs(save_dir, exist_ok=True)
	
	saved = 0
	for i, sample in enumerate(data_load.data):
		label = data_load.labels[i]
		
		# Create synthetic version by adding noise to original coordinates
		original_coords = np.column_stack([sample['x'], sample['y'], sample['z']])
		
		# Add noise to coordinates
		noise = np.random.normal(0, noise_factor * np.std(original_coords, axis=0), 
		                        size=original_coords.shape)
		synthetic_coords = original_coords + noise
		
		# Create 3D plot
		fig = plt.figure(figsize=(10, 8))
		ax = fig.add_subplot(111, projection='3d')
		
		if kind == 'line':
			# Plot original in blue
			ax.plot(original_coords[:, 0], original_coords[:, 1], original_coords[:, 2], 
			       'b-', linewidth=2, alpha=0.7, label='Original')
			# Plot synthetic in red
			ax.plot(synthetic_coords[:, 0], synthetic_coords[:, 1], synthetic_coords[:, 2], 
			       'r--', linewidth=2, alpha=0.8, label=f'Noise={noise_factor}')
		else:
			ax.scatter(original_coords[:, 0], original_coords[:, 1], original_coords[:, 2], 
			          c='blue', s=20, alpha=0.7, label='Original')
			ax.scatter(synthetic_coords[:, 0], synthetic_coords[:, 1], synthetic_coords[:, 2], 
			          c='red', s=20, alpha=0.8, label=f'Noise={noise_factor}')
		
		ax.set_title(f'3D Trajectory Comparison: {label}\n(Original vs Noise-Augmented)')
		ax.set_xlabel('X')
		ax.set_ylabel('Y')
		ax.set_zlabel('Z')
		ax.legend()
		
		# Save plot
		label_dir = os.path.join(save_dir, str(label).replace(' ', '_'))
		os.makedirs(label_dir, exist_ok=True)
		filename = f"{i:03d}_{label}_3d_noise_{noise_factor}.png"
		out_path = os.path.join(label_dir, filename)
		
		fig.savefig(out_path, dpi=200, bbox_inches='tight')
		print(f"Saved 3D plot to {out_path}")
		plt.close(fig)
		
		saved += 1
	
	print(f"Generated {saved} 3D synthetic plots under '{save_dir}'")
	return saved


def plot_only_synthetic_3d(noise_factor=0.05, n_variations=3, save_dir='noise_only_plots', kind='line'):
	"""Generate 3D plots showing only the synthetic (noisy) versions of each sample.
	
	Args:
		noise_factor: amount of noise to add
		n_variations: number of different noise variations per original sample
		save_dir: directory to save plots
		kind: 'line' or 'scatter'
	"""
	print(f"Creating noise-only 3D plots (noise_factor={noise_factor}, variations={n_variations})...")
	
	# Ensure original data is loaded
	if not data_load.data:
		data_load.process()
	
	if not data_load.data:
		raise RuntimeError("No data available from data_load.process()")
	
	os.makedirs(save_dir, exist_ok=True)
	
	# Set random seed for reproducible noise
	np.random.seed(42)
	
	saved = 0
	for i, sample in enumerate(data_load.data):
		label = data_load.labels[i]
		
		# Get original coordinates
		original_coords = np.column_stack([sample['x'], sample['y'], sample['z']])
		
		# Create multiple noise variations
		for variation in range(n_variations):
			# Generate noise
			noise = np.random.normal(0, noise_factor * np.std(original_coords, axis=0), 
			                        size=original_coords.shape)
			synthetic_coords = original_coords + noise
			
			# Create 3D plot for synthetic data only
			fig = plt.figure(figsize=(8, 6))
			ax = fig.add_subplot(111, projection='3d')
			
			if kind == 'line':
				ax.plot(synthetic_coords[:, 0], synthetic_coords[:, 1], synthetic_coords[:, 2], 
				       'r-', linewidth=2)
			else:
				ax.scatter(synthetic_coords[:, 0], synthetic_coords[:, 1], synthetic_coords[:, 2], 
				          c='red', s=15)
			
			ax.set_title(f'Synthetic {label} (Noise={noise_factor}, Var={variation+1})')
			ax.set_xlabel('X')
			ax.set_ylabel('Y')
			ax.set_zlabel('Z')
			
			# Save plot
			label_dir = os.path.join(save_dir, str(label).replace(' ', '_'))
			os.makedirs(label_dir, exist_ok=True)
			filename = f"{i:03d}_{label}_synthetic_v{variation+1}_noise_{noise_factor}.png"
			out_path = os.path.join(label_dir, filename)
			
			fig.savefig(out_path, dpi=200, bbox_inches='tight')
			plt.close(fig)
			
			saved += 1
	
	print(f"Generated {saved} noise-only 3D plots under '{save_dir}'")
	return saved


if __name__ == '__main__':
	# When run standalone, create synthetic 3D visualizations
	print("Creating original sample plot...")
	out = 'sample_plot.png'
	plot_sample(idx=0, kind='line', save_path=out, show=False)
	
	print("Creating 3D plots for noise-augmented data...")
	# Generate 3D comparison plots (original vs synthetic)
	plot_synthetic_3d(noise_factor=0.08, save_dir='synthetic_3d_plots', kind='line')
	
	print("Creating noise-only 3D plots...")
	# Generate plots showing only synthetic versions
	plot_only_synthetic_3d(noise_factor=0.08, n_variations=2, 
	                      save_dir='noise_only_plots', kind='line')
	
	print("All synthetic 3D visualization plots created!")
