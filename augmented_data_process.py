import pandas as pd
import numpy as np
import os
from sklearn.utils import resample

file_paths = [
    #r"C:\Users\ailab\PycharmProjects\ppg2\train_test_negative_above_threshold_0.csv",
    #r"C:\Users\ailab\PycharmProjects\ppg2\train_test_negative_augmented_0_500.csv",
    #r"C:\Users\ailab\PycharmProjects\ppg2\train_test_negative_below_threshold_0.csv",
    r"C:\Users\ailab\PycharmProjects\ppg2\train_test_positive_no_threshold.csv",
    r"C:\Users\ailab\PycharmProjects\ppg2\train_test_negative_no_threshold.csv"
]

positive_files = [
    "train_test_positive_no_threshold.csv",
]

negative_files = [
    "train_test_negative_no_threshold.csv",
]

X_positive = []
X_negative = []

for file_path in file_paths:
    file_name=os.path.basename(file_path)
    print(f"Processing: {file_name}")

    is_positive=any(pos_file in file_path for pos_file in positive_files)
    is_negative=any(neg_file in file_path for neg_file in negative_files)

    try:
        df=pd.read_csv(file_path)

        if is_negative:
            back_data=df.iloc[:, 0:500].values
            print(f"Back 500 colums (all rows) : {back_data.shape}")
            X_negative.append(back_data)

        else:
            all_data=df.values
            print(f"All data: {all_data.shape}")

            if is_positive:
                X_positive.append(all_data)
                print(f"Added {len(all_data)} samples to positive data")
            else:
                X_negative.append(all_data)
                print(f"Added {len(all_data)} samples to negative data")

    except Exception as e:
        print(f"Error processing {file_path} : {str(e)}")

X_positive_combined = np.vstack(X_positive) if X_positive else np.array([])
X_negative_combined = np.vstack(X_negative) if X_negative else np.array([])

print("\nData Summary:")
print(f"Positive samples: {X_positive_combined.shape}")
print(f"Negative samples: {X_negative_combined.shape}")

# 여기서 레이블을 반대로 지정
y_positive = np.zeros(X_positive_combined.shape[0])  # 긍정 -> 0
y_negative = np.ones(X_negative_combined.shape[0])  # 부정 -> 1

X = np.vstack([X_positive_combined, X_negative_combined])
y = np.concatenate([y_positive, y_negative])

print(f"Combined dataset shape: {X.shape}")
print(f"Labels shape: {y.shape}")

save_path = r"C:\Users\ailab\PycharmProjects\ppg2\processed_data_X"
os.makedirs(save_path, exist_ok=True)

np.save(os.path.join(save_path, r"C:\Users\ailab\PycharmProjects\ppg2\processed_data_X\X_data.npy"), X)
np.save(os.path.join(save_path, r"C:\Users\ailab\PycharmProjects\ppg2\processed_data_X\y_labels.npy"), y)

print(f"\n데이터 저장됨 : {save_path}")
print(f"Positive samples (label 0): {len(y_positive)} ({len(y_positive) / len(y) * 100:.1f}%)")
print(f"Negative samples (label 1): {len(y_negative)} ({len(y_negative) / len(y) * 100:.1f}%)")

# 데이터 시각화
try:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(X_positive_combined[0], bins=30,  range=(-6000, 6000), alpha=0.7)
    plt.title("Sample Positive Data Distribution (Label 0)")

    plt.subplot(1, 2, 2)
    plt.hist(X_negative_combined[0], bins=30, range=(-6000, 6000), alpha=0.7)
    plt.title("Sample Negative Data Distribution (Label 1)")

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "data_distribution_samples.png"))
    print(f"Distribution plot saved to {save_path}")
except:
    print("Couldn't generate distribution plots (matplotlib may be missing)")

X_reshaped = X.reshape(X.shape[0], X.shape[1], 1)
print(f"\nReshaped data for 1D CNN: {X_reshaped.shape}")

# 훈련/테스트 8:2로 나누어 파일 저장
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_reshaped, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training data: {X_train.shape}, {y_train.shape}")
print(f"Testing data: {X_test.shape}, {y_test.shape}")

np.save(os.path.join(save_path, r"C:\Users\ailab\PycharmProjects\ppg2\processed_data_X\X_train.npy"), X_train)
np.save(os.path.join(save_path, r"C:\Users\ailab\PycharmProjects\ppg2\processed_data_X\X_test.npy"), X_test)
np.save(os.path.join(save_path, r"C:\Users\ailab\PycharmProjects\ppg2\processed_data_X\y_train.npy"), y_train)
np.save(os.path.join(save_path, r"C:\Users\ailab\PycharmProjects\ppg2\processed_data_X\y_test.npy"), y_test)

print("Train/test split data saved successfully!")

# 박스플롯: 피크의 평균값 비교
try:
    import heartpy as hp
    import seaborn as sns
    import matplotlib.pyplot as plt


    def extract_peak_means(array_2d, sample_rate=25):
        peak_means = []
        for row in array_2d:
            try:
                wd, _ = hp.process(row, sample_rate=sample_rate)
                peaks = wd['peaklist']
                peak_values = row[peaks]
                if len(peak_values) > 0:
                    peak_means.append(np.mean(peak_values))
                else:
                    peak_means.append(np.nan)
            except:
                peak_means.append(np.nan)
        return peak_means

    # 피크 기반 평균값 추출
    positive_peak_means = extract_peak_means(X_positive_combined)
    negative_peak_means = extract_peak_means(X_negative_combined)

    # 데이터프레임 구성
    df_peaks = pd.DataFrame({
        'Mean Peak Height': positive_peak_means + negative_peak_means,
        'Class': ['Positive'] * len(positive_peak_means) + ['Negative'] * len(negative_peak_means)
    })

    # 박스플롯 시각화
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df_peaks, x='Class', y='Mean Peak Height')
    plt.title('Mean Peak Height per Chunk (Positive vs Negative)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "boxplot_peak_heights.png"), dpi=300)
    print(f"Boxplot of peak heights saved to {save_path}")

except Exception as e:
    print(f"Peak analysis failed: {e}")