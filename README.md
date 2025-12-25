# Big Data – Spark Machine Learning Model Evaluation

## Giới thiệu
Repository này tập trung vào việc **xử lý và phân tích dữ liệu lớn bằng Apache Spark (PySpark)** kết hợp với các mô hình Machine Learning trong Spark ML.

Mục tiêu chính của project:
- Tiền xử lý dữ liệu truy cập người dùng trên môi trường Spark
- Huấn luyện và đánh giá nhiều mô hình Machine Learning
- So sánh hiệu năng các mô hình dựa trên các chỉ số đánh giá chuẩn

Toàn bộ quy trình được xây dựng bằng **Spark DataFrame** và **Spark ML Pipeline**, phù hợp cho các bài toán Big Data.

---

##  Mô tả dữ liệu
Dữ liệu sử dụng là log truy cập người dùng trong hệ thống thương mại điện tử, bao gồm các thuộc tính chính:

- `duration_(secs)` – Thời gian truy cập (giây)
- `bytes` – Dung lượng dữ liệu truyền
- `accessed_from` – Nền tảng truy cập (Chrome, Safari, Android App, iOS App, …)
- `gender` – Giới tính
- `country` – Quốc gia
- `membership` – Loại thành viên
- `pay_method` – Phương thức thanh toán
- `sales` – Biến mục tiêu (0/1)

Quy trình tiền xử lý trên Spark:
- Loại bỏ các cột không cần thiết
- Chuẩn hóa dữ liệu dạng text
- Xử lý giá trị thiếu
- Encode các biến phân loại bằng `StringIndexer` và `OneHotEncoder`

---

## Các mô hình được sử dụng
Các mô hình Machine Learning được huấn luyện bằng **Spark ML** bao gồm:

- **Logistic Regression**
- **Decision Tree**
- **Random Forest**

Mỗi mô hình được xây dựng trong **Spark ML Pipeline** để đảm bảo:
- Tiền xử lý và huấn luyện nhất quán
- Dễ mở rộng cho tập dữ liệu lớn
- So sánh công bằng giữa các mô hình

---

## Chỉ số đánh giá
Hiệu năng của các mô hình được đánh giá bằng các chỉ số:

- **Accuracy**
- **Precision (weighted)**
- **Recall (weighted)**
- **F1-score**

Ngoài ra, project còn so sánh:
- Chia dữ liệu cố định (fixed split 20%)
- Chia dữ liệu ngẫu nhiên nhiều lần (random split 20%)

nhằm đánh giá **độ ổn định và khả năng tổng quát hóa của mô hình** trên dữ liệu lớn.
