# Deep-Learning-in-Real-World-Face-Recognition-

# Tôi đang sử dụng GPU của Kaggle cho dự án này, và đây là liên kết đến notebook của tôi (nếu bạn không truy cập được, có thể do tôi đặt ở chế độ riêng tư):  
[Kaggle Notebook: Q4 Deep Learning in Real-World Face Recognition](https://www.kaggle.com/code/nguyenquyetgiangson/q4-deep-learning-in-real-world-face-recognition-a)

# Giới thiệu: Deep Learning trong Ứng dụng Nhận Diện Khuôn Mặt Thực Tế

Dự án "Deep Learning trong Ứng dụng Nhận Diện Khuôn Mặt Thực Tế" nhằm phát triển một hệ thống nhận diện khuôn mặt thực tiễn sử dụng kiến trúc GhostFaceNet hiện đại (2023) và tập dữ liệu gồm 864 ảnh khuôn mặt thuộc 199 danh tính khác nhau. Mục tiêu chính là xây dựng mô hình có độ chính xác trên 98%, đồng thời mô phỏng các tình huống thêm người dùng mới và đăng nhập.

# DEMO
## Ảnh truy vấn người dùng sử dụng để check-in:
![Query image the user used for check-in](https://raw.githack.com/GiangSon-5/Deep-Learning-in-Real-World-Face-Recognition-/main/images/Query%20image%20the%20user%20used%20for%20check-in.jpg)

## Hệ thống nhận diện ảnh truy vấn khớp với người có danh tính:
![The system recognizes the query image matches the person](https://github.com/GiangSon-5/Deep-Learning-in-Real-World-Face-Recognition-/blob/main/images/The%20system%20recognizes%20the%20query%20image%20matches%20the%20person.jpg)  
![The system recognizes the query image matches the person](https://github.com/GiangSon-5/Deep-Learning-in-Real-World-Face-Recognition-/blob/main/images/The%20system%20recognizes%20the%20query%20image%20matches%20the%20person-2.jpg)

# 1. Các bước thực hiện trong dự án

## Phát triển mô hình nhận diện khuôn mặt

* **Kiến trúc mô hình:** Sử dụng GhostFaceNet (2023), một kiến trúc nhẹ và hiệu quả cho bài toán nhận diện khuôn mặt.
* **Khởi tạo mô hình:** Dùng các trọng số được huấn luyện trước từ thư viện như TensorFlow Hub hoặc PyTorch Hub.

## Chuẩn hóa và huấn luyện

* Tạo các vector embedding từ từng ảnh khuôn mặt qua mô hình.
* Áp dụng chuẩn hóa L2 để cải thiện độ chính xác khi so sánh.

## Mô phỏng thêm người dùng mới

* **Tính toán vector đại diện:** Tính trung bình các embedding của từng danh tính để có vector đại diện.
* **Thêm người dùng:** Mở rộng tập dữ liệu bằng cách thêm ID và ảnh chân dung mới, cập nhật file `data.csv` để lưu thông tin người dùng.

## Mô phỏng đăng nhập người dùng

* **Nhận diện danh tính:** So sánh embedding của ảnh đăng nhập với các vector đại diện trong tập dữ liệu.
* **Kết quả:** Tìm người dùng có độ tương đồng cao nhất (cosine similarity).
* **Xác thực:** Hiển thị tất cả ảnh trong tập dữ liệu của danh tính được nhận diện để xác minh trực quan.

# 2. Công cụ và thư viện sử dụng

* TensorFlow và Keras: Huấn luyện và xây dựng mô hình GhostFaceNet.
* FAISS (Facebook AI Similarity Search): Tăng tốc việc tìm kiếm và so sánh các vector embedding.
* OpenCV: Tiền xử lý ảnh (resize, chuyển đổi màu).
* Scikit-learn: Chuẩn hóa L2 và tính toán các chỉ số đánh giá (ROC, AUC).
* Matplotlib: Vẽ đồ thị ROC để đánh giá mô hình.
* Pandas: Xử lý dữ liệu và thao tác với file CSV.

# 3. Hiệu năng và kết quả

* **Độ chính xác:** Đạt mức 99.5%, vượt xa mục tiêu ban đầu là 98%.
* **Tỉ lệ dương tính thật (TPR):** Đạt 98.8% với tỉ lệ âm tính giả (FPR) thấp.
* **Tăng tốc tìm kiếm:** Tích hợp FAISS với chỉ mục IndexFlatIP (tính sản phẩm trong – Inner Product) giúp giảm thời gian so sánh vector và xử lý ảnh tốt hơn với tập dữ liệu lớn.
* **Mô phỏng thành công:** Hệ thống thể hiện độ chính xác cao trong cả việc thêm danh tính mới và nhận diện lúc đăng nhập.

![Evaluation of model performance](https://raw.githack.com/GiangSon-5/Deep-Learning-in-Real-World-Face-Recognition-/main/images/Evaluation%20of%20model%20performance.jpg)

# 4. Đánh giá mô hình và cải tiến

## Ưu điểm

* GhostFaceNet giảm chi phí tính toán nhờ kiến trúc nhẹ — phù hợp với ứng dụng thực tế.
* FAISS giúp nhận diện nhanh hơn khi áp dụng vào tập dữ liệu lớn.
* Các vector embedding được chuẩn hóa tốt, đảm bảo so sánh chính xác.

## Hạn chế

* Tập dữ liệu chỉ có 864 ảnh — còn nhỏ, hạn chế khả năng tổng quát của mô hình cho các tập lớn hơn.
* Cần mở rộng tập dữ liệu để cải thiện khả năng ứng dụng đa dạng hơn.

## Định hướng cải tiến

* Áp dụng các kỹ thuật tăng cường dữ liệu (data augmentation) để đa dạng ảnh huấn luyện.
* Khám phá thêm các kiến trúc như ArcFace hoặc SphereFace để nâng cao độ chính xác.

# 5. Giải thích các file trong dự án

* **`vn2db.npz`**: Chứa dữ liệu đã tiền xử lý — bao gồm vector embedding, danh tính và thông tin cần thiết khác. Giúp tăng tốc xử lý khi nhận diện hoặc thêm người dùng mới.
* **`GhostFaceNet_W1.3_S1_ArcFace.h5`**: File trọng số đã huấn luyện của mô hình GhostFaceNet, tích hợp hàm mất mát ArcFace — giúp tăng độ chính xác khi phân biệt các khuôn mặt.

# 6. Tích hợp YOLOv5 trong dự án

## Lý do tích hợp YOLOv5

YOLOv5 được sử dụng để phát hiện khuôn mặt trong ảnh trước khi đưa vào mô hình GhostFaceNet để nhận diện. Điều này giúp hệ thống có thể nhận ra khuôn mặt ngay cả khi ảnh có nhiều đối tượng hoặc trong luồng camera thực tế.

## Quy trình tích hợp

* Dùng YOLOv5 để xác định vị trí khuôn mặt trong ảnh hoặc video.
* Crop và chuẩn hóa các khuôn mặt phát hiện được (resize, chuyển đổi màu).
* Đưa khuôn mặt đã xử lý vào GhostFaceNet để thực hiện nhận diện danh tính.

## Lợi ích đạt được

* Nâng cao khả năng xử lý ảnh trong môi trường thực tế, đặc biệt khi khung hình có nhiều đối tượng.
* Tối ưu tốc độ nhận diện khuôn mặt bằng cách kết hợp YOLOv5 và FAISS.

# 7. Trình tự thực hiện dự án (Minh họa bằng hình ảnh)

Dưới đây là sơ đồ trực quan mô tả các bước chính trong dự án nhận diện khuôn mặt thực tế, theo thứ tự từ trái sang phải:

<p align="center">
  <img src="https://github.com/GiangSon-5/Deep-Learning-in-Real-World-Face-Recognition-/blob/main/images/image1.jpg" width="240"/>
  &nbsp;&nbsp;&nbsp;➡️&nbsp;&nbsp;&nbsp;
  <img src="https://github.com/GiangSon-5/Deep-Learning-in-Real-World-Face-Recognition-/blob/main/images/image2.jpg" width="240"/>
  &nbsp;&nbsp;&nbsp;➡️&nbsp;&nbsp;&nbsp;
  <img src="https://github.com/GiangSon-5/Deep-Learning-in-Real-World-Face-Recognition-/blob/main/images/image3.jpg" width="240"/>
  &nbsp;&nbsp;&nbsp;➡️&nbsp;&nbsp;&nbsp;
  <img src="https://github.com/GiangSon-5/Deep-Learning-in-Real-World-Face-Recognition-/blob/main/images/image4.jpg" width="240"/>
</p>

📌 Mỗi ảnh đại diện cho một giai đoạn cụ thể:
1️⃣ **Tiền xử lý và phát hiện khuôn mặt**  
2️⃣ **Trích xuất đặc trưng và embedding**  
3️⃣ **Tìm kiếm qua FAISS và xác minh**  
4️⃣ **Hiển thị kết quả nhận diện**

Bạn có thể thay đổi độ rộng từng ảnh bằng cách chỉnh thông số `width` cho phù hợp với giao diện người xem.

