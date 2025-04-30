import pandas as pd
import numpy as np
import os

# 파일 다시 불러오기 (업로드된 최신 버전 사용)
new_file_path = r"C:\Users\ailab\PycharmProjects\ppg2\train_test_negative_augmented_0.csv"

# 파일 불러오기 및 행/열 개수 확인
df_new = pd.read_csv(new_file_path, header=None)
shape_info = df_new.shape  # (행, 열)

print(shape_info)

# 기존 데이터를 numpy 배열로 변환
data_array = df_new.values  # shape: (1505, 1000)

# 슬라이딩 윈도우 방식으로 500열씩 잘라서 1행으로 쌓기
new_rows = []

for row in data_array:
    for i in range(0, 1000, 500):  # 0~499, 500~999
        new_rows.append(row[i:i+500])

# 결과를 DataFrame으로 변환 (shape: 3010, 500)
new_df = pd.DataFrame(new_rows)

# 결과 저장 (원하면 저장 경로 지정 가능)
new_df_path =r"C:\Users\ailab\PycharmProjects\ppg2\train_test_negative_augmented_0_500.csv"
new_df.to_csv(new_df_path, index=False, header=False)

# 결과 형태 반환
print(new_df.shape)