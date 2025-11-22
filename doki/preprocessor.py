from __future__ import annotations
import os
from pathlib import Path

def preprocess_column6(
    project_root: str | os.PathLike,
    raw_dir: str = "raw_data",
    out_dir: str = "preprocessed_data",
    target_col_index: int = 6,   # 0-based index
    valid_prefix: str = "r",     # keep only lines starting with 'r'
) -> None:
    """
    raw_data/<class>/<file>.txt 를 읽어, 'r'로 시작하는 행만 남기고
    콤마로 분리된 열 중 target_col_index(기본=6)만 추출하여
    preprocessed_data/<class>/<file>.txt 로 저장한다.
    """

    project_root = Path(project_root)
    raw_path = project_root / raw_dir
    out_path = project_root / out_dir
    out_path.mkdir(exist_ok=True)

    # 클래스(하위 디렉토리) 순회
    for class_dir in sorted(p for p in raw_path.iterdir() if p.is_dir()):
        rel_class = class_dir.name
        dst_class_dir = out_path / rel_class
        dst_class_dir.mkdir(parents=True, exist_ok=True)

        # 각 txt 파일 순회
        for txt_file in sorted(class_dir.glob("*.txt")):
            lines_out = []

            with txt_file.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # 'r' 로 시작하는 라인만 유효
                    if not line.startswith(valid_prefix + ","):
                        continue

                    cols = line.split(",")
                    if len(cols) <= target_col_index:
                        continue

                    value = cols[target_col_index].strip()
                    if value:
                        lines_out.append(value)

            # 결과 저장
            dst_file = dst_class_dir / txt_file.name
            with dst_file.open("w", encoding="utf-8") as wf:
                wf.write("\n".join(lines_out))

            print(f"[OK] {txt_file.relative_to(project_root)} -> {dst_file.relative_to(project_root)} "
                  f"({len(lines_out)} lines)")

if __name__ == "__main__":
    # 현재 스크립트 위치 기준으로 프로젝트 루트를 자동 인식
    project_root = Path(__file__).resolve().parent
    print("=" * 70)
    print("■■■ STEP 1: PREPROCESSING STARTED ■■■")
    print("=" * 70)
    preprocess_column6(project_root=project_root)
    print("=" * 70)
    print("■■■ All preprocessing completed successfully! ■■■")
    print("=" * 70)
