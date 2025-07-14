# file: convert_models.py
import tensorflow as tf
import tf2onnx
import os

# --- CẤU HÌNH ---
script_dir = os.path.dirname(os.path.abspath(__file__))

# Đường dẫn cho model nhận dạng (GhostFaceNet)
H5_RECOGNIZER_PATH = os.path.join(script_dir, "data/GhostFaceNet_W1.3_S1_ArcFace.h5")
ONNX_RECOGNIZER_PATH = os.path.join(script_dir, "data/ghostfacenet.onnx")

# Đường dẫn cho model phát hiện (YOLOv5 Face)
# Tải file yolov5s_face_dynamic.h5 thủ công từ URL và đặt vào thư mục data
# URL: https://github.com/leondgarse/Keras_insightface/releases/download/v1.0.0/yolov5s_face_dynamic.h5
H5_DETECTOR_PATH = os.path.join(script_dir, "data/yolov5s_face_dynamic.h5")
ONNX_DETECTOR_PATH = os.path.join(script_dir, "data/yolov5s_face_detector.onnx")

def convert_recognizer():
    """Chuyển đổi model GhostFaceNet."""
    print("\n--- Bắt đầu chuyển đổi GhostFaceNet Recognizer ---")
    try:
        model = tf.keras.models.load_model(H5_RECOGNIZER_PATH, compile=False)
        input_signature = [tf.TensorSpec(shape=(None, 112, 112, 3), dtype=tf.float32, name="input")]
        model_proto, _ = tf2onnx.convert.from_keras(model, input_signature, opset=13)
        with open(ONNX_RECOGNIZER_PATH, "wb") as f:
            f.write(model_proto.SerializeToString())
        print(f"✅ GhostFaceNet Recognizer đã được chuyển đổi thành công -> {ONNX_RECOGNIZER_PATH}")
    except Exception as e:
        print(f"❌ Lỗi khi chuyển đổi GhostFaceNet Recognizer: {e}")

def convert_detector():
    """Chuyển đổi model YOLOv5 Face Detector."""
    print("\n--- Bắt đầu chuyển đổi YOLOv5 Face Detector ---")
    try:
        # Kiểm tra xem file .h5 của detector đã tồn tại chưa
        if not os.path.exists(H5_DETECTOR_PATH):
            print(f"Lỗi: Không tìm thấy file {H5_DETECTOR_PATH}")
            print("Vui lòng tải file từ: https://github.com/leondgarse/Keras_insightface/releases/download/v1.0.0/yolov5s_face_dynamic.h5")
            print("Và đặt nó vào thư mục 'data'.")
            return
            
        model = tf.keras.models.load_model(H5_DETECTOR_PATH, compile=False)
        # YOLOv5 nhận ảnh có kích thước động, được padding thành bội số của 32
        # Chúng ta định nghĩa input với kích thước động
        input_signature = [tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.uint8, name="input")]
        model_proto, _ = tf2onnx.convert.from_keras(model, input_signature, opset=13)
        with open(ONNX_DETECTOR_PATH, "wb") as f:
            f.write(model_proto.SerializeToString())
        print(f"✅ YOLOv5 Face Detector đã được chuyển đổi thành công -> {ONNX_DETECTOR_PATH}")
    except Exception as e:
        print(f"❌ Lỗi khi chuyển đổi YOLOv5 Face Detector: {e}")

if __name__ == "__main__":
    convert_recognizer()
    convert_detector()