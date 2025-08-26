# Deep-Learning-in-Real-World-Face-Recognition-

# Tôi đang sử dụng GPU của Kaggle cho dự án này, và đây là liên kết đến notebook của tôi (nếu bạn không truy cập được, có thể do tôi đặt ở chế độ riêng tư):  
[Kaggle Notebook: Q4 Deep Learning in Real-World Face Recognition](https://www.kaggle.com/code/nguyenquyetgiangson/q4-deep-learning-in-real-world-face-recognition-a)

# Giới thiệu: Deep Learning trong Ứng dụng Nhận Diện Khuôn Mặt Thực Tế

Dự án "Deep Learning trong Ứng dụng Nhận Diện Khuôn Mặt Thực Tế" nhằm phát triển một hệ thống nhận diện khuôn mặt thực tiễn sử dụng kiến trúc GhostFaceNet hiện đại (2023) và tập dữ liệu gồm 864 ảnh khuôn mặt thuộc 199 danh tính khác nhau. Mục tiêu chính là xây dựng mô hình có độ chính xác trên 98%, đồng thời mô phỏng các tình huống thêm người dùng mới và đăng nhập.

# DEMO
## 🌐 Triển khai trực tuyến

Bạn có thể trải nghiệm hệ thống nhận diện khuôn mặt qua giao diện web được triển khai tại Hugging Face Spaces:

👉 [face_detector_ghostfacenet trên Hugging Face](https://huggingface.co/spaces/GiangSon-5/face_detector_ghostfacenet)

## 📷 Ảnh truy vấn ban đầu (Check-in):
![Query image the user used for check-in](https://github.com/GiangSon-5/face_detector_realtime/blob/main/images/demo1.jpg)

## 📝 Đăng ký danh tính mới:
![The system recognizes the query image matches the person](https://github.com/GiangSon-5/face_detector_realtime/blob/main/images/demo2.jpg)  

## ✅ Nhận diện lại ảnh truy vấn:
![The system recognizes the query image matches the person](https://github.com/GiangSon-5/face_detector_realtime/blob/main/images/demo3.jpg)

#  Các bước thực hiện trong dự án
## Trình tự thực hiện dự án (Minh họa bằng hình ảnh)

Dưới đây là sơ đồ trực quan mô tả các bước chính trong dự án nhận diện khuôn mặt thực tế, theo thứ tự từ trái sang phải:

<p align="center">
  <img src="https://github.com/GiangSon-5/face_detector_realtime/blob/main/images/image1.jpg" width="100%"/>
  <br>⬇️<br>
  <img src="https://github.com/GiangSon-5/face_detector_realtime/blob/main/images/image2.jpg" width="100%"/>
  <br>⬇️<br>
  <img src="https://github.com/GiangSon-5/face_detector_realtime/blob/main/images/image3.jpg" width="100%"/>
  <br>⬇️<br>
  <img src="https://github.com/GiangSon-5/face_detector_realtime/blob/main/images/image4.jpg" width="100%"/>
</p>

#  Hiệu năng và kết quả

* **Độ chính xác:** Đạt mức 99.5%, vượt xa mục tiêu ban đầu là 98%.
* **Tỉ lệ dương tính thật (TPR):** Đạt 98.8% với tỉ lệ âm tính giả (FPR) thấp.
* **Tăng tốc tìm kiếm:** Tích hợp FAISS với chỉ mục IndexFlatIP (tính sản phẩm trong – Inner Product) giúp giảm thời gian so sánh vector và xử lý ảnh tốt hơn với tập dữ liệu lớn.
* **Mô phỏng thành công:** Hệ thống thể hiện độ chính xác cao trong cả việc thêm danh tính mới và nhận diện lúc đăng nhập.

![Evaluation of model performance](https://github.com/GiangSon-5/face_detector_realtime/blob/main/images/Evaluation%20of%20model%20performance.jpg)

# 🚀 Hệ thống Nhận diện Khuôn mặt Thực tế

Hệ thống sử dụng kết hợp các thành phần sau:

* **YOLOv5** để phát hiện khuôn mặt  
* **GhostFaceNet** để trích xuất đặc trưng (embedding)  
* **FAISS** để so khớp và truy vấn vector  
* **Gradio** để triển khai giao diện người dùng  

---

## 1. 📌 Pipeline Hệ thống

### 🧪 Giai đoạn 1: Thử nghiệm & Đánh giá (trong Notebook)

* Dữ liệu gốc gồm ảnh chân dung và file `data.csv`
* Phát hiện & căn chỉnh khuôn mặt bằng YOLOv5
* Trích xuất embedding từ khuôn mặt qua GhostFaceNet
* Tính toán vector đại diện (mean embedding cho mỗi danh tính)
* Xây dựng FAISS Index để tìm kiếm nhanh
* Đánh giá mô hình với ROC curve & độ chính xác
* **Kết luận:** Chọn mô hình và ngưỡng (threshold) phù hợp

### 🔄 Giai đoạn 2: Chuyển đổi mô hình sang ONNX

* Chạy script `convert_models.py` để chuyển model Keras (`.h5`) sang ONNX (`.onnx`)
* Tạo hai model ONNX:  
  * YOLOv5 (detector)  
  * GhostFaceNet (recognizer)

### 🌐 Giai đoạn 3: Triển khai Ứng dụng bằng Gradio

* Khởi chạy `app.py`, tải các model ONNX và FAISS index

**Hai luồng chức năng chính:**

* **Đăng ký:**  
  Người dùng tải lên 2 ảnh → trích xuất embedding → cập nhật FAISS index  

* **Đăng nhập:**  
  Người dùng tải lên 1 ảnh → tìm kiếm gần nhất trong index → so sánh điểm giống → trả kết quả

---

## 2. 🧠 Phát triển Mô hình Nhận diện Khuôn mặt

* **Kiến trúc chính:** GhostFaceNet (2023) — nhẹ, tối ưu cho thiết bị thực tế  
* **Trọng số pre-trained:** Tải từ TensorFlow Hub hoặc PyTorch Hub  
* **Embedding:** Vector 512 chiều được chuẩn hóa L2  
* **Loss Function:** ArcFace — cải thiện độ phân biệt giữa các danh tính  

---

## 3. 👥 Mô phỏng Thêm Người dùng Mới

* **Vector đại diện:** Trung bình các embedding của 2 ảnh đăng ký  
* **Lưu trữ:** Ghi danh tính và tên file ảnh vào `data.csv`  
* **Cập nhật hệ thống:** FAISS index được thêm người mới mà không cần retrain  

---

## 4. 🔐 Mô phỏng Đăng nhập Người dùng

* **Truy vấn FAISS:** Tìm vector gần nhất trong index (cosine similarity)  
* **So sánh với ngưỡng:** Nếu điểm tương đồng ≥ threshold → xác nhận thành công  
* **Xác thực trực quan:** Hiển thị ảnh gốc của danh tính gần nhất để đối chiếu  

---

## 5. 🧰 Công cụ & Thư viện Sử dụng

| Thư viện         | Vai trò chính                                       |
|------------------|------------------------------------------------------|
| TensorFlow/Keras | Huấn luyện & chuyển đổi GhostFaceNet                |
| ONNX Runtime     | Triển khai model nhận diện & phát hiện              |
| FAISS            | Truy vấn vector hiệu quả (ann search)               |
| OpenCV           | Tiền xử lý ảnh (crop, resize, BGR → RGB)            |
| Scikit-learn     | Tính toán độ chính xác, ROC, chuẩn hóa              |
| Matplotlib       | Vẽ biểu đồ ROC và trực quan hóa kết quả             |
| Pandas           | Quản lý dữ liệu người dùng với CSV                  |
| Gradio           | Xây dựng giao diện người dùng Web                   |

---

## 6. 📁 Giải thích Các File trong Dự Án

| File / Thư mục                        | Mô tả                                                        |
|--------------------------------------|---------------------------------------------------------------|
| `data.csv`                           | Chứa danh sách người dùng và tên file ảnh                    |
| `images/`                            | Thư mục chứa 864 ảnh khuôn mặt đã được crop từ tập dữ liệu gốc |
| `GhostFaceNet_W1.3_S1_ArcFace.h5`    | Model gốc định dạng Keras                                     |
| `detector.onnx`                      | Model YOLOv5 ONNX để phát hiện khuôn mặt                     |
| `recognizer.onnx`                    | Model GhostFaceNet ONNX để trích xuất vector                 |
| `convert_models.py`                  | Script chuyển model sang ONNX                                 |
| `app.py`                             | File triển khai hệ thống nhận diện bằng Gradio              |

---

## 7. 🤖 Lý Do Tích Hợp YOLOv5

* Nhận diện khuôn mặt chính xác trong ảnh nhiều người hoặc ảnh thực tế (camera)  
* Căn chỉnh khuôn mặt (crop) giúp tăng chất lượng embedding  
* **Tự động hóa toàn bộ:** Người dùng chỉ cần tải ảnh → hệ thống sẽ xử lý và nhận diện  




#  Đánh giá mô hình và cải tiến

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









