# 제스처 분류 알고리즘 정리  
*(circle / horizontal / vertical / diagonal_right / diagonal_left)*

## 1. 전체 파이프라인

각 제스처는 하나의 텍스트 파일(`.txt`)로 주어지며, 내부에는

형태의 3D 좌표가 줄 단위로 저장되어 있다.  
`main.py`에서는 파일 하나에 대해 다음 순서로 분류한다.

1. **원(circle) 여부 검사**
   - `circle_check(file_path)`
   - 원으로 판정되면 문자열 `"circle"` 리턴하고 종료
2. **선(line) 방향 분류**
   - `line_check(file_path)`
   - 아래 네 클래스 중 하나로 분류  
     - `"horizontal"`
     - `"vertical"`
     - `"diagonal_right"`
     - `"diagonal_left"`
-> 100% 분류 완료
