import os
import numpy as np
from utils import *

# 참고 | 파일 경로에 문제가 발생하지 않도록 작성을 해두었으나, 만약 파일 경로가 문제가 있다면 알려주세요

# 잠깐! | 이거 정규화 안하면 작은 원과 큰 원 또는 작은 직선에서 위험하다. 이거 기억하자

# 1. 한 움직임 아래에 있는 txt 파일을 리스트로 추출
MOVEMENT_TYPE = "diagonal_left" # circle | diagonal_left | diagonal_right | horizontal | vertical 
PATH = f"../augmented_data/{MOVEMENT_TYPE}"
file_list = os.listdir(PATH)

for file_name in file_list:
    x_list, y_list, z_list = [], [], []
    with open(f"../augmented_data/{MOVEMENT_TYPE}/{file_name}", "r") as f :
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
    print("[Title] Horizontal vs. Vertical")
    print("[DEBUG] x_list elem 개수: ", len(x_list), "y_list elem 개수: ", len(y_list), \
        "z_list elem 개수: ", len(z_list))    
    
    # 각 좌표의 RANGE 정보를 추출해보자
    print("[INFO] x_range: ", x_range, "y_range: ", y_range, "z_range", z_range)        
    
    # 평가까지 해보는거야 (당연히 Horizontal 과 Vertical의 경우를 나누는 경우로 들어가야 하지)
    if MOVEMENT_TYPE == "horizontal" or MOVEMENT_TYPE == "vertical" :
        if (z_range < x_range) and (z_range < y_range) :        
            print("[CON] Horizontal")
            first_result = "horizontal"
        elif (y_range < x_range) and (y_range < z_range) :
            print("[CON] Vertical")
            first_result = "vertical"
        else :
            print("Oh no... Something is going wrong 😅")
            
        if first_result == MOVEMENT_TYPE :
            print("[Horizontal vs. Vertical] ✅ Correct!")
        else : 
            print("[Horizontal vs. Vertical] ❌ Wrong!")              
    else :
        print("[CON] Horizontal 또는 Vertical이 아니기에 생략!")
        
    result = ""
    
    print("=============================================================")
    # breakpoint() # 이 Breakpoint는 x, y, z의 Range 확인을 위함입니다.
    
    # 3. Diagonal Left 또는 Right는 x의 range가 가장 작은 녀석이다. 

    """
    Left 와 Right를 비교할 수 있는 기준은 dz/dy 가 Positive 이면 Diagonal Left이고,
    Negtative 이면 Diagonal Right 이라는 특징을 발견하였으니, 이를 활용해본다.

    그래서, 두 경우의 dz/dy의 값을 구해보도록 하자.
    이게 가능하려면 내가 그린 이차 함수 형태의 y와 z 좌표가 나와야 한다는 가정이 필연적으로 지켜져야 한다!
    """   

    # 가운데 인덱스를 뽑아내자.
    print("[Title] Diagnoal Left VS. Diagonal Right")
    middle_y_idx = len(y_list) // 2
    middle_z_idx = len(z_list) // 2
    print("[INFO] y의 중앙 좌표: ", middle_y_idx, "z의 중앙 좌표: ", middle_z_idx)
    
    # 가운데 인덱스의 값을 뽑아내자
    middle_y_val = y_list[middle_y_idx]
    middle_z_val = z_list[middle_z_idx]
    print("[INFO] y 위치 중앙값: ", middle_y_val, "z 위치 중앙값: ", middle_z_val)
    
    slope_y = middle_y_val - y_list[0]
    slope_z = middle_z_val - z_list[0]      
        
    # 우리가 실제로 확인해야 하는 dz/dy (여기가 핵심이긴 혀)
    slope = slope_z / slope_y       
    if slope > 0 :
        print("[CON] Slope가 Positive이다. 따라서, 이는 Diagonal Left이다.")
        second_result = "diagonal_left"
    else: 
        print("[CON] Slope가 Negative이다. 따라서, 이는 Diagonal Right이다.")    
        second_result = "diagonal_right"
    
    # 정답 여부를 확인해부자고
    if MOVEMENT_TYPE == "diagonal_left" or MOVEMENT_TYPE == "diagonal_right" :
        if second_result == MOVEMENT_TYPE :
            print("[diagonal left vs. diagonal right] ✅ Correct!")
        else :
            print("[diagonal left vs. diagonal right] ❌ Wrong!")
    
    second_result = ""
    print("=============================================================")
    
    # 4. Circle과 Otherwise를 구분하기 위한 방법을 탐구하자
    """ 
    사실 이게 Machine 기준으로는 가장 앞에서 진행이 되어야 함. 
    그러나, 난이도가 가장 어려울 것이라고 판단을 하고 이걸 가장 마지막에 두었음
    """  
    ratio = pca_analysis(x_list=x_list, y_list=y_list, z_list=z_list)
    print("[INFO] Value of Ratio: ", ratio)
    
    RATIO_THRESHOLD=0.5    
    if ratio > RATIO_THRESHOLD :
        third_result = "circle"
        print("[CON] This is Circular Trajectory!")
    else :
        third_result = "linear"
        print("[CON] This is Linear Trajectory!")
        
    if third_result == "circle" and MOVEMENT_TYPE == "circle" :
        print("[circle vs. otherwise] ✅ Correct!")
    elif third_result == "linear" and (
        MOVEMENT_TYPE == "diagonal_left" or 
        MOVEMENT_TYPE == "diagonal_right" or
        MOVEMENT_TYPE == "horizontal" or
        MOVEMENT_TYPE == "vertical"
    ):
        print("[circle vs. otherwise] ✅ Correct!")
    else :
        print("[circle vs. otherwise] ❌ Wrong!")
    
    # 아니 이게 curvature를 구해봤는데 도움이 1도 안되네..
    points = np.column_stack((x_list, y_list, z_list))        
    
    
    

    
    # # 일단 평균을 구하고 말이야.
    # x_mean = compute_mean(x_list)
    # y_mean = compute_mean(y_list)
    # z_mean = compute_mean(z_list)
    # print("[INFO] X좌표의 평균: ", x_mean, "Y좌표의 평균: ", y_mean, "Z좌표의 평균: ", z_mean)
    
    # # 평균과 각 점으로 부터의 거리를 모두 구해보면
    # """
    # 여기서는 평균과 모든 점 사이의 거리를 총합하여 평균점의 위치가 점들과 어느 정도의 차이가 나는지를
    # 확인해서 Circle과 나머지를 구분할 수 있도록 하였습니다.
    # """
    # x_dist = compute_distance(x_list, x_mean)
    # y_dist = compute_distance(y_list, y_mean)
    # z_dist = compute_distance(z_list, z_mean)
    # print("[INFO] X의 Distance: ", x_dist, "Y의 Distance: ", y_dist, "Z의 Distance: ", z_dist)
    
    
    breakpoint()