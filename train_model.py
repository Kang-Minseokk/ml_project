import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import export_text
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from load_and_feature import build_dataset

# ---------------------------------------------------------
# 폴더(디렉토리)가 없으면 자동으로 만들어주는 함수
# → 예: ./results 폴더 없을 때, 결과 저장하려면 필요함
# ---------------------------------------------------------
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


# ---------------------------------------------------------
# 정확도(accuracy)를 막대 그래프로 저장하는 함수
# 입력: accuracy 값(0~1), 저장할 파일 경로
# ---------------------------------------------------------
def plot_accuracy(accuracy, save_path):
    plt.figure(figsize=(6, 4))
    plt.bar(['Accuracy'], [accuracy], color='skyblue')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy')
    plt.savefig(save_path)
    plt.close()


# ---------------------------------------------------------
# classification_report (precision/recall/f1/support)를
# 표(table) 형태로 이미지로 저장하는 함수
# → report는 classification_report(..., output_dict=True) 결과
# ---------------------------------------------------------
def plot_classification_report(report, save_path):
    df_report = pd.DataFrame(report).transpose()
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.axis('tight')
    ax.axis('off')

    table = ax.table(
        cellText=df_report.values,
        colLabels=df_report.columns,
        rowLabels=df_report.index,
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(df_report.columns))))

    plt.title('Classification Report')
    plt.savefig(save_path)
    plt.close()


# ---------------------------------------------------------
# 클래스별 precision / recall / f1-score를
# 막대 그래프로 시각화하는 함수
# ---------------------------------------------------------
def plot_metrics_bar(report, save_path):
    metrics = ['precision', 'recall', 'f1-score']
    df_report = pd.DataFrame(report).transpose()
    df_report = df_report[df_report.index != 'accuracy']

    x = np.arange(len(df_report.index))
    width = 0.2

    plt.figure(figsize=(10, 6))
    for i, metric in enumerate(metrics):
        plt.bar(x + i * width, df_report[metric], width, label=metric)

    plt.xticks(x + width, df_report.index, rotation=45)
    plt.ylabel('Score')
    plt.title('Metrics by Class')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# ---------------------------------------------------------
# trajectory(좌표들)를 2D 또는 3D로 그려서 저장하는 함수
# coords: (N,3) 형태의 XYZ 좌표
# ---------------------------------------------------------
def plot_trajectory(coords, save_path, plot_3d=False):
    if plot_3d:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], label='Trajectory')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title('3D Trajectory')
    else:
        plt.figure()
        plt.plot(coords[:, 0], coords[:, 1], label='Trajectory')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('2D Trajectory')

    plt.legend()
    plt.savefig(save_path)
    plt.close()


# ---------------------------------------------------------
# feature들 간의 관계를 산점도로 시각화
# 예: straightness vs curvature_mean
# ---------------------------------------------------------
def plot_scatter(df, x_feature, y_feature, save_path):
    plt.figure(figsize=(8, 6))

    for label in df['label'].unique():
        subset = df[df['label'] == label]
        plt.scatter(subset[x_feature], subset[y_feature], label=label, alpha=0.7)

    plt.xlabel(x_feature)
    plt.ylabel(y_feature)
    plt.title(f'{x_feature} vs {y_feature}')
    plt.legend()
    plt.savefig(save_path)
    plt.close()


# ---------------------------------------------------------
# raw_data 폴더에서 데이터를 읽어서 DataFrame(df) 생성  
# ---------------------------------------------------------
# 현재 스크립트가 있는 디렉토리를 기준으로 경로 설정
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "raw_data")

df = build_dataset(data_path)

print("전체 데이터 개수:", len(df))
print(f"특성 개수: {len(df.columns) - 2}개 (diagonal 구분 개선)")  # label, file 제외
print(f"특성 이름: {list(df.drop(columns=['label', 'file']).columns)}")
print("\n클래스별 데이터 개수:")
print(df['label'].value_counts())

if df.empty:
    raise ValueError("The dataset is empty. Please check the data files in the augmented_data folder.")

# X = feature들만 사용 / y = 정답(label) / files = 파일명
X = df.drop(columns=["label", "file"])
y = df["label"]
files = df["file"]

# train/test 분리 (80% / 20%) - 파일명도 함께 분리
X_train, X_test, y_train, y_test, files_train, files_test = train_test_split(
    X, y, files, test_size=0.2, random_state=0
)

# ---------------------------------------------------------
# RandomForest 모델 정의 및 학습 (diagonal 구분 최적화)
# ---------------------------------------------------------
model = RandomForestClassifier(
    n_estimators=150,    # 트리 개수 최적화
    max_depth=20,        # 과적합 방지
    min_samples_split=5, # 일반화 향상
    min_samples_leaf=2,  # 과적합 방지
    random_state=42      # 재현성
)
model.fit(X_train, y_train)

# 결과 저장 폴더 생성
output_dir = os.path.join(script_dir, 'results')
ensure_dir(output_dir)

# ---------------------------------------------------------
# 0) 학습된 모델 저장 (12개 특성)
# ---------------------------------------------------------
model_path = os.path.join(output_dir, 'trained_model.pkl')
joblib.dump(model, model_path)
print(f"모델 저장 완료: {model_path}")

# 현재 폴더에도 복사 (gonggigye.py용)
current_model_path = os.path.join(script_dir, 'trained_model.pkl')
joblib.dump(model, current_model_path)
print(f"현재 폴더 모델 저장: {current_model_path}")

# ---------------------------------------------------------
# 1) 정확도 저장
# ---------------------------------------------------------
accuracy = model.score(X_test, y_test)
plot_accuracy(accuracy, os.path.join(output_dir, 'accuracy.png'))

print(f"\n=== Model Accuracy ===\n{accuracy:.4f}\n")

# ---------------------------------------------------------
# 2) Classification Report 출력 + 이미지 저장
# ---------------------------------------------------------
y_pred = model.predict(X_test)

# 텍스트 버전 출력 (터미널용)
report_text = classification_report(y_test, y_pred, target_names=model.classes_)
print("\n===== Classification Report (Text) =====\n")
print(report_text)
print("========================================\n")

# ---------------------------------------------------------
# 틀린 예측 분석 - 구체적인 파일명 출력
# ---------------------------------------------------------
wrong_mask = y_test != y_pred
wrong_files = files_test[wrong_mask]
wrong_actual = y_test[wrong_mask]
wrong_predicted = y_pred[wrong_mask]

print("🔍 틀린 예측 파일들:")
print("="*50)
if len(wrong_files) == 0:
    print("🎉 모든 예측이 정확합니다!")
else:
    for i, (file, actual, predicted) in enumerate(zip(wrong_files, wrong_actual, wrong_predicted)):
        print(f"{i+1}. 파일: {file}")
        print(f"   실제: {actual} → 예측: {predicted}")
        print()
print("="*50)

# 이미지 저장용 dict
report = classification_report(y_test, y_pred, target_names=model.classes_, output_dict=True)

plot_classification_report(report, os.path.join(output_dir, 'classification_report.png'))
plot_metrics_bar(report, os.path.join(output_dir, 'metrics_bar.png'))

# ---------------------------------------------------------
# 3) Trajectory 예시 1개 가져와서 궤적 이미지 저장
# ---------------------------------------------------------
from load_and_feature import load_xyz_from_txt

sample_file = df.iloc[0]['file']
sample_label = df.iloc[0]['label']
sample_path = os.path.join(script_dir, "augmented_data", sample_label, sample_file)

sample_coords = load_xyz_from_txt(sample_path)

plot_trajectory(sample_coords, os.path.join(output_dir, 'trajectory_2d.png'), plot_3d=False)
plot_trajectory(sample_coords, os.path.join(output_dir, 'trajectory_3d.png'), plot_3d=True)

# ---------------------------------------------------------
# 4) scatter plot 생성 (straightness vs curvature_mean)
# ---------------------------------------------------------
plot_scatter(df, 'straightness', 'curvature_mean', os.path.join(output_dir, 'scatter_straightness_curvature.png'))

# ---------------------------------------------------------
# 5) Confusion Matrix (혼동 행렬) 이미지 저장
# ---------------------------------------------------------
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
disp.figure_.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
plt.close()

# ---------------------------------------------------------
# 랜덤포레스트 안의 첫 번째 트리 규칙 출력 (설명가능성 핵심)
# ---------------------------------------------------------
from sklearn.tree import export_text

print("\n============================")
print("RandomForest 모델 정보")
print("============================")
print(f"총 트리 개수: {len(model.estimators_)}")
print(f"사용된 특성 개수: {model.n_features_in_}")
print(f"클래스 개수: {len(model.classes_)}")

# 각 트리의 깊이 정보
tree_depths = [tree.tree_.max_depth for tree in model.estimators_]
print(f"\n 트리 깊이 통계:")
print(f"   최소 깊이: {min(tree_depths)}")
print(f"   최대 깊이: {max(tree_depths)}")
print(f"   평균 깊이: {sum(tree_depths)/len(tree_depths):.2f}")

# 첫 번째 트리의 상세 정보
first_tree = model.estimators_[0]
print(f"\n 첫 번째 트리 (Tree 0) 상세:")
print(f"   깊이: {first_tree.tree_.max_depth}")
print(f"   노드 개수: {first_tree.tree_.node_count}")
print(f"   잎 노드 개수: {first_tree.tree_.n_leaves}")

print("\n============================")
print("Decision Tree Rules (Tree 0)")
print("============================\n")

tree_rules = export_text(
    model.estimators_[0],
    feature_names=list(X.columns),
    decimals=2,
    show_weights=False
)

# 숫자 클래스를 문자열 라벨로 매핑
class_mapping = {i: label for i, label in enumerate(model.classes_)}

for num, name in class_mapping.items():
    tree_rules = tree_rules.replace(f"class: {float(num)}", f"class: {name}")

print(tree_rules)

# ---------------------------------------------------------
# 훈련된 모델 저장 (재사용을 위해)
# ---------------------------------------------------------
import joblib

model_save_path = os.path.join(output_dir, 'trained_model.pkl')
joblib.dump(model, model_save_path)
print(f"\n 모델 저장 완료: {model_save_path}")

print(f"\n 모든 결과가 {output_dir} 폴더에 저장되었습니다!")
print("생성된 파일들:")
print("   - accuracy.png (정확도 그래프)")
print("   - classification_report.png (성능 리포트)")
print("   - confusion_matrix.png (혼동 행렬)")
print("   - trajectory_2d.png, trajectory_3d.png (예시 궤적)")
print("   - scatter_straightness_curvature.png (특성 분포)")
print("   - trained_model.pkl (훈련된 모델)")