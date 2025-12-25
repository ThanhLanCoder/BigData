import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import seaborn as sn

data = pd.read_csv('/PhanTichNDSpark/data/dataRowOld.csv')
data.plot(kind='density', subplots=True, layout=(3, 3), sharex=False)
plt.show()

numeric_data = data.select_dtypes(include=['number'])

# Vẽ biểu đồ Box Plot cho tất cả các cột số
plt.figure(figsize=(12, 6))  # chỉnh kích thước biểu đồ
numeric_data.boxplot(rot=45)  # xoay nhãn trục X cho dễ nhìn
plt.title('Biểu đồ Box Plot cho các biến số')
plt.ylabel('Giá trị')
plt.grid(True)
plt.tight_layout()
plt.show()

# Vẽ Scatter Matrix Plot
plt.figure(figsize=(10, 10))
scatter_matrix(numeric_data, figsize=(12, 12), diagonal='hist', alpha=0.7)
plt.suptitle("Scatter Matrix Plot cho các biến số")
plt.show()

selected_cols = ['duration_(secs)', 'bytes', 'age', 'sales']
sn.pairplot(data[selected_cols])
plt.suptitle("Pairplot cho các biến liên quan đến hành vi mua hàng", y=1.02)
plt.show()
