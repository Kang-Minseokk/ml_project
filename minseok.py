import os


# 1. 한 움직임 아래에 있는 txt 파일을 리스트로 추출
MOVEMENT_TYPE = "diagonal_left" # circle | diagonal_left | diagonal_right | horizontal | vertical 
PATH = f"./augmented_data/{MOVEMENT_TYPE}"
file_list = os.listdir(PATH)

for file_name in file_list:
    x_list, y_list, z_list = [], [], []
    with open(f"./augmented_data/{MOVEMENT_TYPE}/{file_name}", "r") as f :
        while True:
            line = f.readline()
            if not line :
                break
            x, y, z = line.split('/')
            x_list.append(int(x))
            y_list.append(int(y))
            z_list.append(int(z))
    
# 2. 각 파일의 좌표 데이터 통계를 내보자
    max_x, max_y, max_z = max(x_list), max(y_list), max(z_list)
    min_x, min_y, min_z = min(x_list), min(y_list), min(z_list)
    x_range, y_range, z_range = max_x - min_x, max_y - min_y, max_z - min_z
    # 디버깅용 원소 개수를 출력해보자     
    print("[DEBUG] x_list elem 개수: ", len(x_list), "y_list elem 개수: ", len(y_list), \
        "z_list elem 개수: ", len(z_list))    
    
    # 각 좌표의 RANGE 정보를 추출해보자
    print("[INFO] x_range: ", x_range, "y_range: ", y_range, "z_range", z_range)
    
    breakpoint()
    
    
    
    
    