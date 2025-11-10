# 자, 우선은 데이터를 가져와서 시각화를 해보자
import pandas as pd
import os
import matplotlib.pyplot as plt
import math

category = ["circle", "diagonal_left", "diagonal_right", "horizontal", "vertical"]
diff_x_list = []
diff_y_list = []
diff_z_list = []
file_names = []

for cat in category:    
    data_list = os.listdir(cat)
    print(data_list)
    for data_name in data_list:
        file_names.append(data_name)
        
        df = pd.read_csv(f"{cat}/{data_name}", delimiter=",")
        row_num = df.shape[0]        
        
        # 첫번째 점과 마지막 점의 diff를 우선 구해주자  
        row = df.iloc[0]        
        print("This is row: ", row) 
        first_point = df.iloc[0, 6] 
        print(first_point)
        print("====")       
        
        # 첫번째 점의 x, y, z
        fx, fy, fz = map(int, first_point.split("/"))
        
        last_point = df.iloc[row_num-1, 6]        
        
        # 마지막 점의 x, y, z
        lx, ly, lz = map(int, last_point.split("/"))
        
        # diff 구하자
        diff_x = abs(fx - lx)
        diff_y = abs(fy - ly)
        diff_z = abs(fz - lz)
        
        euclidian_distance = math.sqrt(diff_x**2 + diff_y**2 + diff_z**2)
        print(f"{cat}-{data_name}: ", euclidian_distance)
        
        diff_x_list.append(diff_x)
        diff_y_list.append(diff_y)
        diff_z_list.append(diff_z)        

    # ✅ 시각화
    # plt.figure(figsize=(10, 6))

    # plt.plot(diff_x_list, marker='o', label='diff_x')
    # plt.plot(diff_y_list, marker='o', label='diff_y')
    # plt.plot(diff_z_list, marker='o', label='diff_z')

    # plt.title("Diff results")
    # plt.xlabel("Sample index")
    # plt.ylabel("Difference")
    # plt.legend()
    # plt.grid(True)
    # plt.xticks(range(len(file_names)), file_names, rotation=45)

    # plt.tight_layout()
    # plt.show()        