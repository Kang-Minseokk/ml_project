import os
from load_and_feature import load_xyz_from_txt
from gonggigye import Sungcheol, Minseok, Jaeeun
from collections import Counter

# [NOTE] STEP1: 입력 데이터를 전처리 후 processed_data에 저장을 해줍니다.
PATH = "data"
file_list = os.listdir(PATH)        
# print(f"[INFO] 입력 파일 개수 : {len(file_list)}")

for file_name in file_list:    
    file_path = os.path.join(PATH, file_name)
    preprocessed_data = load_xyz_from_txt(file_path)
    
    # 앞에 processed_data라는 prefix를 붙여줍니다.
    filepath = os.path.join("processed_data", f"processed_{file_name.split('.')[0]}.txt")
    
    with open(filepath, "w") as f:
        for (x, y, z) in preprocessed_data:            
            f.write(f"{x} {y} {z}\n")

# [NOTE] STEP2: 각 모델에 넣어서 결과를 출력해봅시다.
dir_path = "processed_data"
file_path_list = []
for name in os.listdir(dir_path):
    file_path_list.append(os.path.join(dir_path, name))

for file_path in file_path_list:
    # 성철님 모델 동작
    sung_result = Sungcheol.circle_check(file_path)
    if sung_result != 'circle':
        sung_result = Sungcheol.line_check(file_path)    
    print(f"[INFO] Sungcheol 모델 출력값: {sung_result}")
    
    # 민석씨 모델 동작
    min_result = Minseok.minseok_circle(file_path)
    x_list, y_list, z_list = Minseok.make_list(file_path)
    x_range, y_range, z_range = Minseok.check_range(x_list, y_list, z_list)
    if (min_result != 'circle') and (x_range < y_range and x_range < z_range):        
        min_result = Minseok.minseok_check2(file_path)    
    if min_result == 'linear':
        min_result = Minseok.minseok_check1(file_path)    
    print(f"[INFO] Minseok 모델 출력값: {min_result}")
    
    # 재은님 모델 동작
    jae_result = Jaeeun.predict_trajectory(file_path)
    
    print(f"[INFO] Jaeeun 모델 출력값: {jae_result}")

# [NOTE] STEP3: 각 모델의 결과를 Voting 해줍시다
    results = [sung_result, min_result, jae_result]

    vote = Counter(results).most_common(1)[0][0]

    print(f"[INFO]⭐️ 최종 Voting 결과 ⭐️ : {vote}")
    print("=" * 50)
    print("\n")
