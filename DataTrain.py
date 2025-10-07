# ==============================
# File: ThongKeMoTa.py
# Mục đích: Thống kê mô tả dữ liệu và trực quan hóa cột Fare
# ==============================

import pandas as pd
import matplotlib.pyplot as plt

# --- Đọc dữ liệu ---
df = pd.read_csv('D:/BigData/Buoi2/train.csv')
print("=== Thông tin ban đầu của DataFrame ===")
print(df.info())

# --- Lọc các cột dạng số ---
df_number = df.select_dtypes(include=['number'])

# --- Điền giá trị khuyết ---
df_filled = df_number.interpolate()

# --- Loại bỏ cột PassengerId ---
df_final = df_filled.drop(columns=['PassengerId'])
print("\n=== Thông tin sau khi xử lý dữ liệu ===")
print(df_final.info())

# --- Các thống kê mô tả ---
data_count = df_final.count()
data_mean = df_final.mean()
data_median = df_final.median()
data_var = df_final.var()
data_std = df_final.std()
data_min = df_final.min()
data_quantile = df_final.apply(lambda x: x.quantile([0.25, 0.5, 0.75]))
data_iqr = df_final.apply(lambda x: x.quantile(0.75) - x.quantile(0.25))
data_max = df_final.max()

# --- Tạo bảng tổng hợp ---
summary = {
    'count': data_count,
    'mean': data_mean,
    'median': data_median,
    'var': data_var,
    'std': data_std,
    'min': data_min,
    'q1': data_quantile.loc[0.25],
    'q2': data_quantile.loc[0.5],
    'q3': data_quantile.loc[0.75],
    'iqr': data_iqr,
    'max': data_max
}

df_summary = pd.DataFrame(summary).T

# --- In kết quả ---
print("\n=== Bảng tổng hợp các chỉ số mô tả ===\n")
print(df_summary)

print("\n=== Bảng thống kê mô tả (describe) ===\n")
print(df_final.describe())

# --- Biểu đồ Histogram cho cột Fare ---
plt.figure(figsize=(10, 6))
plt.hist(df_final['Fare'], bins=50, edgecolor='black')
plt.xlabel('Fare')
plt.ylabel('Frequency')
plt.title('Histogram of Fare')
plt.grid(False)
plt.show()

# --- Biểu đồ Histogram cho toàn bộ dữ liệu ---
plt.figure(figsize=(15, 10))
df_final.hist(bins=50, edgecolor='black', grid=False)
plt.tight_layout()
plt.show()

# --- Biểu đồ Boxplot cho các cột dạng số ---
plt.figure(figsize=(15, 10))
df_final.plot(kind='box', subplots=True, layout=(2, 4), sharex=False, sharey=False)
plt.tight_layout()
plt.show()
