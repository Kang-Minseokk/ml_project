from line_circle_classification import change_numpy, sc_pca, circle_classfication
from line_classification import change_numpy_yz, fit_line, plus_determine

import os

def circle_check(file_path):
    data = change_numpy(file_path)
    transformed_data = sc_pca(data)
    is_circle = circle_classfication(transformed_data)
    if is_circle:
        return 'circle'

def line_check(file_path):
    data = change_numpy_yz(file_path)
    a = fit_line(data)
    if abs(a) < 0.3:
        return 'horizontal'
    elif abs(a) > 4:
        return 'vertical'
    else:
        a, b = plus_determine(data)
        if a < 0:
            return 'diagonal_right'
        else:
            return 'diagonal_left'
    
def main():
    circle_path = "/Users/parksung-cheol/Desktop/ml_project-doki/augmented_data/circle"
    diagonal_left_path = "/Users/parksung-cheol/Desktop/ml_project-doki/augmented_data/diagonal_left"
    diagonal_right_path = "/Users/parksung-cheol/Desktop/ml_project-doki/augmented_data/diagonal_right"
    horizontal_path = "/Users/parksung-cheol/Desktop/ml_project-doki/augmented_data/horizontal"
    vertical_path = "/Users/parksung-cheol/Desktop/ml_project-doki/augmented_data/vertical"

    paths = [circle_path, diagonal_left_path, diagonal_right_path, horizontal_path, vertical_path]
    for path in paths:
        file_list = sorted(os.listdir(path))
        for filename in file_list:
            if filename.endswith(".txt"):
                file_path = os.path.join(path, filename)
                shape = circle_check(file_path)
                if shape is None:
                    shape = line_check(file_path)
                print(f"{filename}: {shape}")

if __name__ == "__main__": main()