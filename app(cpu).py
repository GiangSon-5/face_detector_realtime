# file: app.py (phiên bản TensorFlow + YoloV5FaceDetector)
import gradio as gr
import numpy as np
import cv2
import os
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import normalize
from tqdm import tqdm
import faiss

# Import lớp YoloV5FaceDetector từ file của bạn
from face_detector import YoloV5FaceDetector

# --- CẤU HÌNH VÀ BIẾN TOÀN CỤC ---
script_dir = os.path.dirname(os.path.abspath(__file__))

# Đường dẫn tới các tài nguyên
RECOGNIZER_MODEL_PATH = os.path.join(script_dir, "data/GhostFaceNet_W1.3_S1_ArcFace.h5")
INITIAL_DATA_CSV = os.path.join(script_dir, "data/data_new/data.csv")
INITIAL_IMAGE_ROOT = os.path.join(script_dir, "data/data_new/images")
UPLOAD_DIR = os.path.join(script_dir, "uploads")

if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# --- KHỞI TẠO CÁC MODEL ---

# 1. Khởi tạo YoloV5FaceDetector
# Lớp này sẽ tự động tải model yolov5s_face_dynamic.h5 nếu không tìm thấy
print("Đang tải model Face Detector (YOLOv5)...")
detector = YoloV5FaceDetector()
print("Tải Face Detector thành công!")

# 2. Khởi tạo model nhận dạng GhostFaceNet
try:
    print("Đang tải model Face Recognizer (GhostFaceNet)...")
    recognizer_model = tf.keras.models.load_model(RECOGNIZER_MODEL_PATH, compile=False)
    # Tạo interface để trích xuất embedding
    model_interf = lambda imms: recognizer_model((imms - 127.5) * 0.0078125, training=False).numpy()
    print("Tải Face Recognizer thành công!")
except Exception as e:
    print(f"Lỗi: Không thể tải model tại '{RECOGNIZER_MODEL_PATH}'. Lỗi: {e}")
    exit()

# 3. Kiểm tra Faiss GPU
try:
    FAISS_USE_GPU = faiss.get_num_gpus() > 0
    if FAISS_USE_GPU: print(f"Faiss: Đã tìm thấy {faiss.get_num_gpus()} GPU. Sẽ sử dụng GPU.")
    else: print("Faiss: Không tìm thấy GPU, sẽ sử dụng CPU.")
except AttributeError:
    FAISS_USE_GPU = False; print("Faiss: Phiên bản faiss-cpu được cài đặt, sẽ sử dụng CPU.")

# Biến toàn cục
db_data = {"all_embs": None, "all_classes": None}
user_data = {}
faiss_index = None
next_user_id = 0
faiss_id_map = []

# --- CÁC HÀM XỬ LÝ CHUẨN ---

def detect_and_get_embedding(image_np):
    """
    Quy trình chuẩn: Dùng YOLO để phát hiện và căn chỉnh, sau đó dùng GhostFaceNet để embedding.
    """
    # 1. Phát hiện và căn chỉnh khuôn mặt bằng YoloV5FaceDetector
    # image_np cần ở định dạng RGB
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    _, _, _, aligned_faces = detector.detect_in_image(image_rgb)
    
    if aligned_faces.shape[0] == 0:
        print("Cảnh báo: Không tìm thấy khuôn mặt trong ảnh.")
        return None

    # Lấy khuôn mặt đầu tiên (đã được căn chỉnh và resize về 112x112)
    face_112x112 = aligned_faces[0]
    
    # 2. Trích xuất embedding từ khuôn mặt đã căn chỉnh
    face_expanded = np.expand_dims(face_112x112, axis=0)
    emb = model_interf(face_expanded)
    
    return normalize(emb.astype("float32"))[0]

# Các hàm logic chính (initialize, update_faiss, register, login)
# sẽ gọi hàm `detect_and_get_embedding` mới này. Code của chúng không thay đổi nhiều.
def update_faiss_index():
    """
    Tính toán lại các vector đại diện từ đầu và cập nhật index Faiss.
    Tự động sử dụng GPU nếu có.
    """
    global faiss_index, faiss_id_map
    print("Đang cập nhật lại Faiss index...")

    # 1. Lấy tất cả các ID đã có (cả gốc và mới) và tạo map
    all_known_ids = []
    if db_data["all_classes"] is not None:
        all_known_ids = np.unique(db_data["all_classes"]).tolist()
    all_known_ids.extend(list(user_data.keys()))
    faiss_id_map = sorted(list(set(all_known_ids))) # Map từ index của Faiss -> ID thật

    if not faiss_id_map:
        print("Không có người dùng nào để index.")
        faiss_index = None
        return

    # 2. Tính toán vector đại diện
    representative_embs = []
    for identity_id in tqdm(faiss_id_map, desc="Tạo vector đại diện"):
        embs_for_user = []
        if identity_id in user_data: # Là người dùng mới
            image_paths = user_data[identity_id]
            for p in image_paths:
                emb = detect_and_get_embedding(cv2.imread(p))
                if emb is not None: embs_for_user.append(emb)
        else: # Là người dùng trong dataset gốc
            indices = np.where(db_data["all_classes"] == identity_id)[0]
            embs_for_user = db_data["all_embs"][indices]
        
        # ----- DÒNG ĐÃ SỬA LỖI -----
        if len(embs_for_user) == 0: continue
        # ---------------------------
        
        sum_emb = np.sum(embs_for_user, axis=0)
        rep_emb = normalize([sum_emb])[0]
        representative_embs.append(rep_emb)
    
    if not representative_embs:
        print("Không có embedding hợp lệ để tạo index."); faiss_index = None; return
        
    # 3. Xây dựng Faiss Index (GPU hoặc CPU)
    representative_embs_np = np.array(representative_embs).astype('float32')
    dimension = representative_embs_np.shape[1]

    if FAISS_USE_GPU:
        res = faiss.StandardGpuResources()
        cpu_index = faiss.IndexFlatIP(dimension)
        gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        faiss_index = gpu_index
    else:
        faiss_index = faiss.IndexFlatIP(dimension)

    faiss_index.add(representative_embs_np)
    print(f"Cập nhật Faiss index thành công với {faiss_index.ntotal} người dùng.")


def initialize_system():
    global db_data, next_user_id
    print("Đang khởi tạo hệ thống từ dataset ban đầu...")
    try:
        df = pd.read_csv(INITIAL_DATA_CSV)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {INITIAL_DATA_CSV}. Bỏ qua."); update_faiss_index(); return

    embs, classes = [], []
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Trích xuất embedding gốc"):
        img_path = os.path.join(INITIAL_IMAGE_ROOT, row['image'])
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            emb = detect_and_get_embedding(img)
            if emb is not None:
                embs.append(emb)
                classes.append(row['label'])

    if embs:
        db_data["all_embs"] = np.array(embs)
        db_data["all_classes"] = np.array(classes)
        next_user_id = db_data["all_classes"].max() + 1
    else:
        next_user_id = 0
    update_faiss_index()
    print(f"Khởi tạo hoàn tất. ID tiếp theo: {next_user_id}")


def register_face(img1, img2):
    global user_data, next_user_id
    if img1 is None or img2 is None:
        return "Lỗi: Vui lòng tải lên cả hai ảnh.", None

    if detect_and_get_embedding(img1) is None or detect_and_get_embedding(img2) is None:
        return "Lỗi: Không tìm thấy khuôn mặt trong một hoặc cả hai ảnh. Vui lòng thử lại với ảnh khác.", None

    registered_id = next_user_id
    img1_path = os.path.join(UPLOAD_DIR, f"{registered_id}_1.jpg"); cv2.imwrite(img1_path, cv2.cvtColor(img1, cv2.COLOR_RGB2BGR))
    img2_path = os.path.join(UPLOAD_DIR, f"{registered_id}_2.jpg"); cv2.imwrite(img2_path, cv2.cvtColor(img2, cv2.COLOR_RGB2BGR))
    
    user_data[registered_id] = [img1_path, img2_path]
    next_user_id += 1
    update_faiss_index()
    
    return f" Đăng ký thành công! ID của bạn là: {registered_id}", user_data[registered_id]


def login_face(login_img):
    if login_img is None: return "Lỗi: Vui lòng tải ảnh lên.", None, None
    if faiss_index is None or faiss_index.ntotal == 0: return "Lỗi: Hệ thống chưa có người dùng.", None, None

    login_emb = detect_and_get_embedding(login_img)
    if login_emb is None:
        return " Không nhận dạng được: không tìm thấy khuôn mặt trong ảnh đăng nhập.", None, None
    
    login_emb_np = np.expand_dims(login_emb.astype('float32'), axis=0)
    D, I = faiss_index.search(login_emb_np, 1)
    faiss_match_index = I[0][0]; best_match_score = D[0][0]
    matched_id_in_db = faiss_id_map[faiss_match_index]
    
    print(f"Faiss search: ID gần nhất {matched_id_in_db} với độ tương đồng {best_match_score:.4f}")
    
    threshold = 0.5
    
    if best_match_score < threshold:
        return f" Không nhận dạng được. (Score: {best_match_score:.4f} < {threshold})", login_img, None
    else:
        if matched_id_in_db in user_data:
            gallery_output = user_data.get(matched_id_in_db, [])
        else:
            df = pd.read_csv(INITIAL_DATA_CSV)
            image_files = df[df['label'] == matched_id_in_db]['image'].tolist()
            gallery_output = [os.path.join(INITIAL_IMAGE_ROOT, fname) for fname in image_files]
            
        message = f" Đăng nhập thành công! Nhận dạng là người dùng có ID: {matched_id_in_db} (Score: {best_match_score:.4f})"
        return message, login_img, gallery_output

# --- GIAO DIỆN GRADIO ---
with gr.Blocks(theme=gr.themes.Soft(), title="Hệ thống Nhận dạng Khuôn mặt") as demo:
    gr.Markdown("# Ứng dụng Nhận dạng Khuôn mặt (TensorFlow + YOLOv5 Detector)")
    gr.Markdown("Quy trình chuẩn: **Phát hiện khuôn mặt bằng YOLOv5** -> Trích xuất Embedding -> Nhận dạng.")
    
    with gr.Tabs():
        # Giao diện không thay đổi
        with gr.TabItem(" Đăng ký"):
            gr.Markdown("## Đăng ký người dùng mới"); gr.Markdown("Tải lên 2 ảnh chân dung rõ nét của bạn để hệ thống học.")
            with gr.Row():
                img_reg_1 = gr.Image(label="Ảnh 1", type="numpy", height=224, width=224)
                img_reg_2 = gr.Image(label="Ảnh 2", type="numpy", height=224, width=224)
            btn_register = gr.Button("🚀 Bắt đầu Đăng ký", variant="primary")
            output_register_status = gr.Textbox(label="Trạng thái", interactive=False)
            output_register_gallery = gr.Gallery(label="Ảnh đã đăng ký", object_fit="contain", height="auto")

        with gr.TabItem(" Đăng nhập"):
            gr.Markdown("## Đăng nhập vào hệ thống"); gr.Markdown("Tải lên 1 ảnh của bạn để xác thực.")
            img_login = gr.Image(label="Ảnh Đăng nhập", type="numpy", height=224, width=224)
            btn_login = gr.Button("🛰️ Đăng nhập", variant="primary")
            output_login_status = gr.Textbox(label="Kết quả Nhận dạng", interactive=False)
            with gr.Row():
                output_query_img = gr.Image(label="Ảnh bạn đã dùng để đăng nhập", type="numpy", height=224, width=224)
                output_login_gallery = gr.Gallery(label="Ảnh của người dùng được nhận dạng", object_fit="contain", height="224")

    btn_register.click(fn=register_face, inputs=[img_reg_1, img_reg_2], outputs=[output_register_status, output_register_gallery])
    btn_login.click(fn=login_face, inputs=[img_login], outputs=[output_login_status, output_query_img, output_login_gallery])

if __name__ == "__main__":
    initialize_system()
    demo.launch(debug=True)