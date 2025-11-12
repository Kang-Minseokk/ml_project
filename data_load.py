import os
import sys
import pandas as pd

data = []
labels = []

base_dir = r"C:/Projects/2025_ml_project/1st_data"

def process():
    if not os.path.exists(base_dir):
        print(f"ERROR: base_dir does not exist: {base_dir}")
        return

    file_count = 0
    skipped = 0
    # Walk directory tree to support an extra top-level folder or nested label folders
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".txt"):
                path = os.path.join(root, file)
                file_count += 1
                # derive label from the parent folder name (the immediate folder containing the file)
                label = os.path.basename(root)
                try:
                    df = pd.read_csv(path, header=None)
                except Exception as e:
                    # 다른 구분자로 읽어보기
                    try:
                        df = pd.read_csv(path, header=None, delim_whitespace=True, engine='python')
                        print(f"Warning: used delim_whitespace for {path}")
                    except Exception as e2:
                        print(f"Failed to read {path}: {e}; {e2}")
                        skipped += 1
                        continue

                if df.shape[1] <= 6:
                    print(f"Skipping {path}: not enough columns ({df.shape[1]})")
                    skipped += 1
                    continue

                col = df.iloc[:, 6]
                col = col.dropna()

                def parse_cell(x):
                    try:
                        parts = str(x).split('/')
                        return list(map(float, parts))
                    except Exception:
                        return None

                coords = col.apply(parse_cell)
                coords = coords.dropna().tolist()
                if not coords:
                    print(f"No valid coords in {path}")
                    skipped += 1
                    continue

                coords_df = pd.DataFrame(coords, columns=['x', 'y', 'z'])
                data.append(coords_df)
                labels.append(label)

    print(f"Processed files: {file_count}, skipped: {skipped}, data items: {len(data)}")
    if data:
        print("Sample label counts:", {l: labels.count(l) for l in set(labels)})
        print("First data sample head:")
        print(data[0].head())


if __name__ == '__main__':
    process()