import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import data_load


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


if __name__ == '__main__':
	# When run standalone, save a PNG instead of blocking on a GUI show.
	out = 'sample_plot.png'
	plot_sample(idx=0, kind='line', save_path=out, show=False)


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
