# visualize_preprocessed.py
from __future__ import annotations
import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # GUI 없이 저장만 할 때 안전
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (3D 등록용)

def read_xyz_lines(path: Path):
    xs, ys, zs = [], [], []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # "x/y/z" 형태
            try:
                x, y, z = line.split("/")
                xs.append(float(x)); ys.append(float(y)); zs.append(float(z))
            except Exception:
                # 형식이 다른 라인은 건너뜀
                continue
    return xs, ys, zs

def set_axes_equal(ax):
    # 3D에서 축 스케일 균등하게 맞추기 (원/원궤가 타원처럼 보이지 않게)
    import numpy as np
    xlim = ax.get_xlim3d(); ylim = ax.get_ylim3d(); zlim = ax.get_zlim3d()
    xmid = (xlim[0] + xlim[1]) / 2.0
    ymid = (ylim[0] + ylim[1]) / 2.0
    zmid = (zlim[0] + zlim[1]) / 2.0
    radius = max(
        (xlim[1] - xlim[0]) / 2.0,
        (ylim[1] - ylim[0]) / 2.0,
        (zlim[1] - zlim[0]) / 2.0,
    )
    ax.set_xlim3d([xmid - radius, xmid + radius])
    ax.set_ylim3d([ymid - radius, ymid + radius])
    ax.set_zlim3d([zmid - radius, zmid + radius])

def plot_and_save(xs, ys, zs, title: str, out_png: Path):
    fig = plt.figure(figsize=(6, 6), dpi=160)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(xs, ys, zs, linewidth=1.5, color='red')
    ax.set_title(title)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    
    ax.view_init(elev=25, azim=40) # 보기 좋은 각도(elev=25, azim=40) (elev : z축으로의 시점 변화, azim : x,y 평면으로의 시점 변화)
    try:
        set_axes_equal(ax)
    except Exception:
        pass
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)

def main():
    project_root = Path(__file__).resolve().parent
    src_root = project_root / "augmented_data"
    dst_root = project_root / "visualization_augmented_data"
    if not src_root.exists():
        raise FileNotFoundError(f"Not found: {src_root}")

    total = 0
    for class_dir in sorted(p for p in src_root.iterdir() if p.is_dir()):
        category = class_dir.name
        out_dir = dst_root / category
        out_dir.mkdir(parents=True, exist_ok=True)

        for txt in sorted(class_dir.glob("*.txt")):
            xs, ys, zs = read_xyz_lines(txt)
            if len(xs) < 2:
                print(f"[SKIP] {txt} (insufficient points)")
                continue
            out_png = out_dir / (txt.stem + ".png")
            plot_and_save(xs, ys, zs, category, out_png)
            total += 1
            print(f"[OK] {txt.relative_to(project_root)} -> {out_png.relative_to(project_root)}")

    print(f"\nDone. Saved {total} plots under {dst_root}.")

if __name__ == "__main__":
    main()
    
