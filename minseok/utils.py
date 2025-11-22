import numpy as np

def compute_mean(target) :
    return sum(target) / len(target)

def compute_distance(target, mean_value) :
    result = 0
    for val in target :
        diff = val - mean_value
        result += diff
    return result        

def pca_analysis(x_list, y_list, z_list):
    """
    PCA 기반으로 궤적 형태를 판별합니다.
    
    return:
    - "line"   : 거의 직선 궤적
    - "circle" : 평면상 곡선(원 가능성 높음)
    """
    points = np.column_stack([x_list, y_list, z_list])
    mean = points.mean(axis=0)
    centered = points - mean

    U, S, Vt = np.linalg.svd(centered)

    # 방향성 분산 체크
    ratio = S[1] / S[0]        
    
    return ratio        

def compute_curvature(points):
    curvatures = []
    for i in range(1, len(points)-1):
        p1 = points[i-1]
        p2 = points[i]
        p3 = points[i+1]
        
        a = np.linalg.norm(p2 - p1)
        b = np.linalg.norm(p3 - p2)
        c = np.linalg.norm(p3 - p1)

        s = (a+b+c)/2
        area = max(s*(s-a)*(s-b)*(s-c), 0)**0.5

        if a*b*c != 0:
            curvature = 4*area/(a*b*c)
            curvatures.append(curvature)

    return np.std(curvatures)