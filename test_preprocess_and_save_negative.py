import numpy as np
import heartpy as hp
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# 파일 경로
file1 = r"C:\Users\ailab\PycharmProjects\ppg2\testdata_negative.csv"  # 이미 500개씩 분할된 파일
file2 = r"C:\Users\ailab\PycharmProjects\ppgStudy\concated_ppg_data_negative.txt"  # 1D raw 시계열

# 파일 불러오기
data1 = np.loadtxt(file1, delimiter=",")  # shape: (n, 500)
data2_raw = np.loadtxt(file2)  # shape: (m,)

# data2는 1D이므로 reshape 필요
def reshape_vector_to_matrix(vector, row_size=500):
    n = len(vector)
    stride = row_size  # 겹치지 않게
    segments = []

    for i in range(0, n - row_size + 1, stride):
        window = vector[i:i + row_size]
        if len(window) < row_size:
            window = np.pad(window, (0, row_size - len(window)), mode='constant', constant_values=np.nan)
        segments.append(window)
    return np.array(segments)

data2 = reshape_vector_to_matrix(data2_raw, 500)

# 두 데이터 병합
merged_data = np.vstack((data1, data2))  # shape: (n+m, 500)

# 유효 청크 및 피크 처리
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
                    avg_peak_value = sum_peak / len(real_peaks) #각 청크당 유효한 피크의 평균값
                    avg_peak_array.append(avg_peak_value) #그 평균값을 모은 배열
                    valid_segments.append(valid_data)

    except Exception as e:
        print(f"{i}번째 청크 처리 실패: {str(e)}")

# 평균, 표준편차 및 threshold 계산
avg_peak_array = np.array(avg_peak_array)
valid_segments = np.array(valid_segments)

std_var = 0
neg_mean = np.mean(avg_peak_array) # 각 청크에서 유효한 피크의 평균을 구하고, 이를 통해 모든 청크의 피크 평균값을 구한 값
neg_std = np.std(avg_peak_array)
threshold = neg_mean + std_var * neg_std

# threshold 저장
threshold_csv_path = f'threshold_value_{std_var}.csv'
np.savetxt(threshold_csv_path, [threshold], delimiter=',')
print(f"Threshold 값이 CSV 파일로 저장됨: {threshold_csv_path}")

# 조건 필터링
high_peak_indices = np.where(avg_peak_array < threshold)[0]
selected_segments = valid_segments[high_peak_indices]
selected_peak_values = avg_peak_array[high_peak_indices]

# 결과 저장
if len(selected_segments) > 0:
    output_path = r"C:\Users\ailab\PycharmProjects\ppg2\train_test_negative_below_threshold.csv"
    np.savetxt(output_path, selected_segments, delimiter=',')

    print(f"Negative Peak Array 평균: {neg_mean}")
    print(f"Negative Peak Array 표준편차: {neg_std}")
    print(f"조건에 맞는 청크 수: {len(selected_segments)}/{len(merged_data)}")
    print(f"CSV 파일이 성공적으로 저장됨: {output_path}")
    print(f"저장된 청크 수: {len(selected_segments)}")
    print(f"각 청크 길이: {selected_segments[0].shape[0]}")
    print(f"데이터 shape: {selected_segments.shape}")
else:
    print("조건에 맞는 청크가 없음.")