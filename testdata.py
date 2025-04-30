import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import heartpy as hp
from keras.models import load_model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

save_dir=r"C:\Users\ailab\PycharmProjects\ppg2"

# 파일 경로 설정
file_paths = [
    r"C:\Users\ailab\PycharmProjects\ppg2\VR_test_data\jh_left\1681815686266\ppg_green.txt",
    r"C:\Users\ailab\PycharmProjects\ppg2\VR_test_data\jh_right\1681261790949\ppg_green.txt",
    r"C:\Users\ailab\PycharmProjects\ppg2\VR_test_data\m1_left\1675931125936\ppg_green.txt",
    r"C:\Users\ailab\PycharmProjects\ppg2\VR_test_data\m1_right\1681822657751\ppg_green.txt",
    r"C:\Users\ailab\PycharmProjects\ppg2\VR_test_data\m2_left\1681269276864\ppg_green.txt",
    r"C:\Users\ailab\PycharmProjects\ppg2\VR_test_data\m2_right\1675932659426\ppg_green.txt",
    r"C:\Users\ailab\PycharmProjects\ppg2\VR_test_data\m3_left\1675932257870\ppg_green.txt",
    r"C:\Users\ailab\PycharmProjects\ppg2\VR_test_data\m3_right\1681823785425\ppg_green.txt",
    r"C:\Users\ailab\PycharmProjects\ppg2\VR_test_data\m4_left\1675933819438\ppg_green.txt",
    r"C:\Users\ailab\PycharmProjects\ppg2\VR_test_data\m4_right\1681270427012\ppg_green.txt",
    r"C:\Users\ailab\PycharmProjects\ppg2\VR_test_data\w1_left\1681824836513\ppg_green.txt",
    r"C:\Users\ailab\PycharmProjects\ppg2\VR_test_data\w1_right\1675933307027\ppg_green.txt",
    r"C:\Users\ailab\PycharmProjects\ppg2\VR_test_data\w2_left\1675934782377\ppg_green.txt",
    r"C:\Users\ailab\PycharmProjects\ppg2\VR_test_data\w2_right\1681271399482\ppg_green.txt"
]

# PPG 파일 불러오기
def process_ppg_and_save_raw(file_paths, save_dir, row_size=500):
    os.makedirs(save_dir, exist_ok=True)

    positive_segments = []  # 앞 절반 (label 0)
    negative_segments = []  # 뒤 절반 (label 1)

    for file_counter, file_path in enumerate(file_paths):
        print(f"[{file_counter+1}/{len(file_paths)}] 파일 처리 중: {file_path}")

        if not os.path.exists(file_path):
            print(f"파일 없음: {file_path}")
            continue

        try:
            with open(file_path, 'r') as file:
                lines = file.readlines()

            # PPG 값 추출
            file_data = [int(line.strip().split()[1]) for line in lines[15:-1]]
            n = len(file_data)
            stride = row_size
            total_chunks = (n - row_size) // stride + 1 #전체 청크 개수
            half_chunks = total_chunks // 2 # 그 절반 개수

            # 앞 절반 0, 뒤 절반 1
            for i in range(total_chunks):
                start_idx = i * stride
                end_idx = start_idx + row_size
                window_data = file_data[start_idx:end_idx]

                if len(window_data) < row_size:
                    window_data = np.pad(window_data, (0, row_size - len(window_data)), mode='constant', constant_values=0)

                label = 0 if i < half_chunks else 1
                row = list(window_data)

                if label == 0:
                    positive_segments.append(row)
                else:
                    negative_segments.append(row)

        except Exception as e:
            print(f"파일 처리 중 오류 발생: {e}")

    # CSV 저장
    if positive_segments:
        df_pos = pd.DataFrame(positive_segments)
        df_pos.to_csv(os.path.join(save_dir, 'testdata_positive.csv'), index=False,header=False)
        print(f"Positive 세그먼트 {len(positive_segments)}개 저장 완료") #(106, 500)

    if negative_segments:
        df_neg = pd.DataFrame(negative_segments)
        df_neg.to_csv(os.path.join(save_dir, 'testdata_negative.csv'), index=False,header=False)
        print(f"Negative 세그먼트 {len(negative_segments)}개 저장 완료") #(112, 500)

process_ppg_and_save_raw(file_paths, save_dir)