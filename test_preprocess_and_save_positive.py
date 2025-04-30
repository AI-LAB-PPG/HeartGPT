import numpy as np
import heartpy as hp
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# 파일 경로
file1 = r"C:\Users\ailab\PycharmProjects\ppg2\testdata_positive.csv" #전처리 안한 test 데이터
file2 = r"C:\Users\ailab\PycharmProjects\ppgStudy\concated_ppg_data_positive.txt" # 기존의 train 데이터

# 데이터 로딩
data1 = np.loadtxt(file1, delimiter=",")  # (n, 500)
data2_raw = np.loadtxt(file2)             # (m,)

# data2: 시계열을 500개 단위 청크로 변환
def reshape_vector_to_matrix(vector, row_size=500):
    n = len(vector)
    stride = row_size
    segments = []
    for i in range(0, n - row_size + 1, stride):
        window = vector[i:i + row_size]
        if len(window) < row_size:
            window = np.pad(window, (0, row_size - len(window)), mode='constant', constant_values=np.nan)
        segments.append(window)
    return np.array(segments)

data2 = reshape_vector_to_matrix(data2_raw, 500)

# 병합
merged_data = np.vstack((data1, data2))  # (n + m, 500)

# 청크별 필터링 및 피크 분석
valid_segments = []
avg_peak_array = []

for i, row in enumerate(merged_data):
    try:
        valid_data = row[~np.isnan(row)]
        valid_data = hp.filter_signal(valid_data, cutoff=[0.5, 8], sample_rate=25, order=3, filtertype="bandpass")

        wd, m = hp.process(valid_data, sample_rate=25)
        if not np.max(wd['hr']) > 15000 or np.min(wd['hr']) < -15000:
            if (len(wd['peaklist']) - len(wd['removed_beats'])) > (len(valid_data) / 25) / 2:
                real_peaks = [item for item in wd['peaklist'] if item not in wd['removed_beats']]
                sum_peak = sum(valid_data[index] for index in real_peaks)

                if len(real_peaks) > 0:
                    avg_peak_value = sum_peak / len(real_peaks)
                    avg_peak_array.append(avg_peak_value)
                    valid_segments.append(valid_data)

    except Exception as e:
        print(f"{i}번째 청크 처리 실패: {str(e)}")

# 평균, 표준편차 계산
avg_peak_array = np.array(avg_peak_array)
valid_segments = np.array(valid_segments)

threshold=0
low_peak_indices=np.where(avg_peak_array>threshold)[0]

# threshold 없이 모두 사용
output_path = r"C:\Users\ailab\PycharmProjects\ppg2\train_test_positive_no_threshold.csv"
np.savetxt(output_path, valid_segments, delimiter=',')

print(f"Positive Peak Array 평균: {np.mean(avg_peak_array)}")
print(f"Positive Peak Array 표준편차: {np.std(avg_peak_array)}")
print(f"조건에 맞는 청크 수 : {len(low_peak_indices)}/{len(merged_data)}")
print(f"CSV 파일이 성공적으로 저장됨: {output_path}")
print(f"저장된 청크 수: {len(valid_segments)}")
print(f"각 청크 길이: {valid_segments[0].shape[0]}")
