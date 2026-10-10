# Lab 1 — Supervised Learning

Bài thực hành học có giám sát với bộ dữ liệu **California Housing**, sử dụng Python và Scikit-learn. Code, kết quả và phần giải thích được trình bày trong notebook `lab1_notebook.ipynb`.

Bài lab giải quyết hai bài toán:

- **Hồi quy:** dự đoán giá nhà median của một khu vực.
- **Phân loại:** xác định khu vực thuộc nhóm giá cao (`MedHouseVal >= 2.5`, tương đương 250.000 USD) hay giá thấp (`MedHouseVal < 2.5`).

## Dữ liệu

California Housing có **20.640 mẫu**, **8 đặc trưng đầu vào** và cột mục tiêu `MedHouseVal`. Mỗi mẫu đại diện cho một khu vực, không phải một căn nhà riêng lẻ. Giá trị mục tiêu có đơn vị **100.000 USD**.

Notebook lấy dữ liệu bằng `fetch_california_housing(as_frame=True)`. Lần tải đầu tiên cần kết nối Internet; những lần sau có thể dùng dữ liệu đã lưu trong bộ nhớ đệm trên máy.

## Nội dung notebook

| Phần | Nội dung |
|---|---|
| Part 1 | Kiểm tra môi trường và thư viện |
| Part 2 | Khám phá dữ liệu, vẽ biểu đồ và phân tích tương quan |
| Part 3 | Chia train/test theo tỷ lệ 80/20, tạo đặc trưng và chuẩn hóa dữ liệu |
| Part 4 | Linear Regression, SGDRegressor, Polynomial Regression, Ridge và Lasso |
| Part 5 | K-Nearest Neighbors (KNN), Gaussian Naive Bayes, Decision Tree, Random Forest và Support Vector Machines (SVM) |
| Part 6 | Tóm tắt kết quả và đề xuất cải thiện |

## Các file trong dự án

```text
ProjectML/
├── README.md                   # Giới thiệu và hướng dẫn chạy
├── lab1_notebook.ipynb          # Code, output và nhận xét của bài lab
├── eda_plots.png               # Biểu đồ khám phá dữ liệu
├── learning_curves.png         # Đường cong học của các mô hình hồi quy
├── knn_accuracy_vs_k.png        # Accuracy khi thay đổi k của KNN
└── rf_feature_importance.png    # Độ quan trọng của đặc trưng trong Random Forest
```

## Cách chạy bằng VS Code trên Windows

### 1. Chuẩn bị môi trường

Cài **Python 3.11**, **VS Code** và hai extension **Python**, **Jupyter** trong VS Code. Mở thư mục dự án bằng **File > Open Folder**.

Nếu dự án đã có môi trường `.venv` và các thư viện cần thiết, chuyển sang bước 2. Nếu chạy trên máy mới, mở terminal PowerShell tại thư mục dự án rồi chạy:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install numpy pandas matplotlib seaborn scikit-learn ipykernel
```

Các lệnh trên dùng trực tiếp Python trong `.venv`, không cần kích hoạt môi trường bằng `Activate.ps1`.

### 2. Chọn môi trường cho notebook

1. Mở `lab1_notebook.ipynb`.
2. Bấm **Select Kernel** ở góc trên bên phải.
3. Chọn môi trường Python của dự án: `.venv\Scripts\python.exe`.

### 3. Chạy bài lab

Chọn **Restart Kernel**, sau đó **Run All** để chạy các ô từ đầu đến cuối. Các phần sau sử dụng biến và mô hình được tạo ở phần trước, nên cần chạy đúng thứ tự.

Việc đánh giá chéo và huấn luyện SVM có thể mất vài phút tùy máy. Các biểu đồ được lưu vào thư mục làm việc của notebook. Sau khi chạy xong, lưu notebook để giữ output, bảng kết quả và biểu đồ.

## Môi trường đã dùng

Các phiên bản được ghi nhận trong môi trường của dự án khi viết README:

| Thành phần | Phiên bản |
|---|---|
| Python (kernel notebook) | 3.11.15 |
| NumPy | 2.4.6 |
| Pandas | 3.0.6 |
| Matplotlib | 3.11.2 |
| Seaborn | 0.13.2 |
| Scikit-learn | 1.9.1 |
| ipykernel | 7.4.0 |

Lệnh cài đặt phía trên không cố định phiên bản. Khi sử dụng phiên bản khác, thông báo cảnh báo hoặc kết quả số có thể khác nhẹ.

## Quy trình thực hiện để có kết quả

Bài lab đi theo trình tự: **Đọc dữ liệu → Khám phá dữ liệu → Chia train/test → Tiền xử lý → Huấn luyện mô hình → Dự đoán trên test → Chấm điểm và so sánh**. Hai bài toán hồi quy và phân loại dùng cùng cách chia dữ liệu để các mô hình được đánh giá trên cùng những mẫu kiểm tra.

### 1. Đọc và tìm hiểu dữ liệu

Tải California Housing vào DataFrame, kiểm tra số mẫu, kiểu dữ liệu, giá trị thiếu và thống kê mô tả. Sau đó vẽ phân bố giá nhà, biểu đồ tương quan, quan hệ giữa thu nhập và giá nhà, cùng biểu đồ so sánh các vùng địa lý.

Bước này giúp hiểu dữ liệu trước khi huấn luyện. Kết quả phân tích cho thấy thu nhập (`MedInc`) có tương quan mạnh nhất với giá nhà trong các đặc trưng đầu vào; giá nhà cũng khác nhau giữa các vùng.

### 2. Chia dữ liệu để học và kiểm tra

Tạo 5 nhóm thu nhập từ `MedInc`, rồi chia dữ liệu theo tỷ lệ **80% train — 20% test** bằng cách phân tầng theo nhóm thu nhập, với `random_state=42`:

- **16.512 mẫu train:** dùng để huấn luyện và chọn tham số bằng đánh giá chéo.
- **4.128 mẫu test:** dùng để chấm kết quả dự đoán của các mô hình đã huấn luyện.

Phân tầng giúp tỷ lệ các nhóm thu nhập gần giống nhau giữa hai tập. Cột tạm `income_cat` được xóa sau khi chia và không đưa vào mô hình.

### 3. Chuẩn bị đầu vào cho mô hình

Tạo ba cột theo yêu cầu của đề để có tổng cộng **11 cột đầu vào**:

| Cột tạo thêm | Cách tính trong notebook |
|---|---|
| `rooms_per_household` | Bằng `AveRooms`, vì cột gốc đã là số phòng trung bình mỗi hộ |
| `bedrooms_ratio` | `AveBedrms / AveRooms`, có xử lý trường hợp chia cho 0 |
| `population_per_household` | Bằng `AveOccup`, vì cột gốc đã là số người trung bình mỗi hộ |

Như vậy, hai cột đầu vào mới trùng thông tin với cột có sẵn; `bedrooms_ratio` bổ sung thông tin về tỷ lệ phòng ngủ.

Tiếp theo, dùng `Pipeline` gồm `SimpleImputer(strategy="median")` để xử lý giá trị thiếu nếu có và `StandardScaler()` để chuẩn hóa các cột. Pipeline **chỉ học thông số từ train** bằng `fit_transform()`; test dùng lại thông số đó qua `transform()`. Khi chọn tham số bằng đánh giá chéo, bước tiền xử lý cũng được học lại từ phần huấn luyện của từng lượt.

### 4. Huấn luyện và đánh giá các mô hình hồi quy

Dùng 11 cột đầu vào để dự đoán giá nhà `MedHouseVal`. Thử Linear Regression, SGDRegressor, Polynomial Regression bậc 2, Ridge và Lasso.

Với Ridge, thử `alpha` trong `[0.01, 0.1, 1, 10, 100]` bằng đánh giá chéo 5 lượt trên train và chọn được **`alpha=10`**. Với Linear Regression và Polynomial Regression, vẽ đường cong học để xem sai lệch thay đổi thế nào khi tăng dữ liệu huấn luyện.

Sau khi học bằng `fit()`, mỗi mô hình dùng `predict()` để dự đoán giá trên test. So sánh giá dự đoán với giá thật bằng **RMSE, MAE và R²**, đồng thời đối chiếu điểm train/test để nhận xét khả năng khái quát hóa. RMSE và MAE càng thấp càng tốt; R² càng cao càng tốt.

Trong bảng kết quả test, **Polynomial Regression bậc 2** có RMSE và MAE thấp nhất, R² cao nhất. Đây là căn cứ để ghi mô hình này vào bảng kết quả chính, dù đường cong học cho thấy nó còn chưa ổn định qua đánh giá chéo.

### 5. Huấn luyện và đánh giá các mô hình phân loại

Đổi giá nhà thật thành hai nhãn: **`1` nếu giá từ 250.000 USD trở lên**, **`0` nếu thấp hơn**. Sau đó huấn luyện các mô hình dự đoán nhãn này:

- KNN với `k=5` và KNN với **`k=10`**, được chọn khi thử `k` từ 1 đến 20 bằng đánh giá chéo 5 lượt trên train.
- Gaussian Naive Bayes.
- Decision Tree với `max_depth=5`.
- Random Forest với **100 cây**.
- SVM với hai kernel **RBF** và **Linear**, dùng toàn bộ tập train.

Chấm cả 7 mô hình trên cùng tập test bằng **Accuracy, Precision, Recall và AUC-ROC**. Accuracy tính từ nhãn dự đoán; AUC-ROC dùng xác suất dự đoán của nhóm giá cao. Precision và recall trong bảng tổng hợp cũng tính cho nhóm giá cao (`1`). Do hai nhóm không cân bằng, cần xem các chỉ số cùng nhau thay vì chỉ dựa vào accuracy.

**Random Forest** đạt cả bốn chỉ số cao nhất trong bảng so sánh hiện tại, nên được chọn làm kết quả phân loại chính.

### 6. Tổng hợp và diễn giải kết quả

Lấy các điểm từ bảng hồi quy ở Part 4 và bảng phân loại ở Part 5 để lập bảng bên dưới. MAE hồi quy được nhân với **100.000** để đổi sang USD; giá trị khoảng `0,45666` tương ứng sai lệch tuyệt đối trung bình khoảng **45.666 USD**. Accuracy **89,97%** nghĩa là Random Forest dự đoán đúng khoảng 90 trong 100 mẫu kiểm tra.

Các kết quả này mô tả những mô hình đã thử trên lần chia dữ liệu hiện tại. Part 6 trong notebook tổng hợp nhận xét, giới hạn và hai hướng cải thiện; chưa thực hiện các cải tiến đó trong các con số báo cáo.

## Kết quả chính đã ghi nhận

| Bài toán | Mô hình tốt nhất trên tập kiểm tra hiện tại | Kết quả |
|---|---|---|
| Hồi quy | Polynomial Regression bậc 2 | RMSE: 0,6615; MAE: khoảng 45.666 USD; R²: 0,6642 |
| Phân loại | Random Forest với 100 cây | Accuracy: 89,97%; AUC-ROC: 0,9557 |

RMSE trong bảng dùng đơn vị gốc của mục tiêu là 100.000 USD. MAE đã được đổi sang USD và biểu thị sai lệch tuyệt đối trung bình khi dự đoán giá nhà trung vị của khu vực.

Polynomial Regression cho kết quả tốt nhất trên tập kiểm tra hiện tại nhưng còn dao động qua đánh giá chéo. Random Forest dẫn đầu bảng so sánh phân loại; việc triển khai thực tế cần đánh giá thêm dữ liệu mới, thời gian dự đoán và bộ nhớ.

Các phân tích chi tiết và câu trả lời theo đề nằm trong **ô Markdown của notebook**. README giới thiệu dự án, hướng dẫn cách chạy và tóm tắt quy trình dẫn đến các kết quả trên.
