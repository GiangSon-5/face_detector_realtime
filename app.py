# file: app.py (PyTorch/ONNX)
import gradio as gr
import numpy as np
import cv2
import os
import pandas as pd
from sklearn.preprocessing import normalize
from tqdm import tqdm
import faiss
import onnxruntime as ort
from face_detector import YoloV5FaceDetector  # Import lớp ONNX detector mới

# --- CẤU HÌNH VÀ BIẾN TOÀN CỤC ---
script_dir = os.path.dirname(os.path.abspath(__file__))

ONNX_RECOGNIZER_PATH = os.path.join(script_dir, "data/ghostfacenet.onnx")
ONNX_DETECTOR_PATH = os.path.join(script_dir, "data/yolov5s_face_detector.onnx")
INITIAL_DATA_CSV = os.path.join(script_dir, "data/data_new/data.csv")
INITIAL_IMAGE_ROOT = os.path.join(script_dir, "data/data_new/images")
UPLOAD_DIR = os.path.join(script_dir, "uploads")

if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# --- KHỞI TẠO CÁC MODEL ONNX ---
print("Đang tải các model ONNX...")
detector = YoloV5FaceDetector(model_path=ONNX_DETECTOR_PATH)
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
recognizer_session = ort.InferenceSession(ONNX_RECOGNIZER_PATH, providers=providers)
print(f"Recognizer ONNX đang chạy trên: {recognizer_session.get_providers()}")

try:
    FAISS_USE_GPU = faiss.get_num_gpus() > 0
    if FAISS_USE_GPU:
        print(f"Faiss: Đã tìm thấy {faiss.get_num_gpus()} GPU.")
    else:
        print("Faiss: Sẽ sử dụng CPU.")
except AttributeError:
    FAISS_USE_GPU = False
    print("Faiss: Sẽ sử dụng CPU.")

db_data = {"all_embs": None, "all_classes": None}
user_data = {}
faiss_index = None
next_user_id = 0
faiss_id_map = []


# --- CÁC HÀM XỬ LÝ ---
def detect_and_get_embedding(image_np):
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    _, _, _, aligned_faces = detector.detect_in_image(image_rgb)
    if aligned_faces.shape[0] == 0:
        return None

    face_112x112 = aligned_faces[0]
    face_normalized = (face_112x112 - 127.5) * 0.0078125
    face_expanded = np.expand_dims(face_normalized, axis=0).astype(np.float32)

    input_name = recognizer_session.get_inputs()[0].name
    output_name = recognizer_session.get_outputs()[0].name
    emb = recognizer_session.run([output_name], {input_name: face_expanded})[0]
    return normalize(emb.astype("float32"))[0]


def update_faiss_index():
    global faiss_index, faiss_id_map
    print("Đang cập nhật lại Faiss index...")
    faiss_id_map = sorted(
        list(
            set(
                list(user_data.keys())
                + (
                    np.unique(db_data["all_classes"]).tolist()
                    if db_data["all_classes"] is not None
                    else []
                )
            )
        )
    )

    if not faiss_id_map:
        faiss_index = None
        return

    representative_embs = []
    for identity_id in tqdm(faiss_id_map, desc="Tạo vector đại diện"):
        embs_for_user = []
        if identity_id in user_data:
            for p in user_data[identity_id]:
                emb = detect_and_get_embedding(cv2.imread(p))
                if emb is not None:
                    embs_for_user.append(emb)
        else:
            embs_for_user = db_data["all_embs"][
                np.where(db_data["all_classes"] == identity_id)[0]
            ]

        if len(embs_for_user) == 0:
            continue
        representative_embs.append(normalize([np.sum(embs_for_user, axis=0)])[0])

    if not representative_embs:
        faiss_index = None
        return

    rep_embs_np = np.array(representative_embs).astype("float32")
    dimension = rep_embs_np.shape[1]
    if FAISS_USE_GPU:
        res = faiss.StandardGpuResources()
        cpu_index = faiss.IndexFlatIP(dimension)
        faiss_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
    else:
        faiss_index = faiss.IndexFlatIP(dimension)
    faiss_index.add(rep_embs_np)
    print(f"Cập nhật Faiss index thành công với {faiss_index.ntotal} người dùng.")


def initialize_system():
    global db_data, next_user_id
    print("Đang khởi tạo hệ thống...")
    try:
        df = pd.read_csv(INITIAL_DATA_CSV)
    except FileNotFoundError:
        update_faiss_index()
        return

    embs, classes = [], []
    for _, row in tqdm(
        df.iterrows(), total=df.shape[0], desc="Trích xuất embedding gốc"
    ):
        img_path = os.path.join(INITIAL_IMAGE_ROOT, row["image"])
        if os.path.exists(img_path):
            emb = detect_and_get_embedding(cv2.imread(img_path))
            if emb is not None:
                embs.append(emb)
                classes.append(row["label"])

    if embs:
        db_data = {"all_embs": np.array(embs), "all_classes": np.array(classes)}
        next_user_id = db_data["all_classes"].max() + 1
    else:
        next_user_id = 0
    update_faiss_index()
    print(f"Khởi tạo hoàn tất. ID tiếp theo: {next_user_id}")


def register_face(img1, img2):
    global user_data, next_user_id
    if img1 is None or img2 is None:
        return "Lỗi: Vui lòng tải lên cả hai ảnh.", None
    if (
        detect_and_get_embedding(cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)) is None
        or detect_and_get_embedding(cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)) is None
    ):
        return "Lỗi: Không tìm thấy khuôn mặt trong một hoặc cả hai ảnh.", None

    registered_id = next_user_id
    img1_path = os.path.join(UPLOAD_DIR, f"{registered_id}_1.jpg")
    cv2.imwrite(img1_path, cv2.cvtColor(img1, cv2.COLOR_RGB2BGR))
    img2_path = os.path.join(UPLOAD_DIR, f"{registered_id}_2.jpg")
    cv2.imwrite(img2_path, cv2.cvtColor(img2, cv2.COLOR_RGB2BGR))

    user_data[registered_id] = [img1_path, img2_path]
    next_user_id += 1
    update_faiss_index()
    return (
        f" Đăng ký thành công! ID của bạn là: {registered_id}",
        user_data[registered_id],
    )


def login_face(login_img):
    if login_img is None:
        return "Lỗi: Vui lòng tải ảnh lên.", None, None
    if faiss_index is None or faiss_index.ntotal == 0:
        return "Lỗi: Hệ thống chưa có người dùng.", None, None

    login_emb = detect_and_get_embedding(cv2.cvtColor(login_img, cv2.COLOR_RGB2BGR))
    if login_emb is None:
        return " Không nhận dạng được: không tìm thấy khuôn mặt.", None, None

    D, I = faiss_index.search(np.expand_dims(login_emb.astype("float32"), axis=0), 1)
    matched_id = faiss_id_map[I[0][0]]
    score = D[0][0]
    print(f"Faiss search: ID gần nhất {matched_id} với độ tương đồng {score:.4f}")

    if score < 0.5:
        return f" Không nhận dạng được. (Score: {score:.4f})", login_img, None
    else:
        gallery_output = user_data.get(matched_id, [])
        if not gallery_output:
            df = pd.read_csv(INITIAL_DATA_CSV)
            gallery_output = [
                os.path.join(INITIAL_IMAGE_ROOT, f)
                for f in df[df["label"] == matched_id]["image"]
            ]
        return (
            f" Đăng nhập thành công! ID: {matched_id} (Score: {score:.4f})",
            login_img,
            gallery_output,
        )


# --- GIAO DIỆN GRADIO ---
with gr.Blocks(theme=gr.themes.Soft(), title="Hệ thống Nhận dạng Khuôn mặt") as demo:
    gr.Markdown("# Ứng dụng Nhận dạng Khuôn mặt ")

    with gr.Tabs():
        with gr.TabItem(" Đăng ký"):
            gr.Markdown("## Đăng ký người dùng mới")
            with gr.Row():
                img_reg_1 = gr.Image(label="Ảnh 1", type="numpy", height=224, width=224)
                img_reg_2 = gr.Image(label="Ảnh 2", type="numpy", height=224, width=224)
            btn_register = gr.Button("🚀 Bắt đầu Đăng ký", variant="primary")
            output_register_status = gr.Textbox(label="Trạng thái", interactive=False)
            output_register_gallery = gr.Gallery(
                label="Ảnh đã đăng ký", object_fit="contain", height="auto"
            )

        with gr.TabItem(" Đăng nhập"):
            gr.Markdown("## Đăng nhập vào hệ thống")
            img_login = gr.Image(
                label="Ảnh Đăng nhập", type="numpy", height=224, width=224
            )
            btn_login = gr.Button(" Đăng nhập", variant="primary")
            output_login_status = gr.Textbox(
                label="Kết quả Nhận dạng", interactive=False
            )
            with gr.Row():
                output_query_img = gr.Image(
                    label="Ảnh bạn đã dùng", type="numpy", height=224, width=224
                )
                output_login_gallery = gr.Gallery(
                    label="Ảnh người dùng trong CSDL",
                    object_fit="contain",
                    height="224",
                )

    btn_register.click(
        fn=register_face,
        inputs=[img_reg_1, img_reg_2],
        outputs=[output_register_status, output_register_gallery],
    )
    btn_login.click(
        fn=login_face,
        inputs=[img_login],
        outputs=[output_login_status, output_query_img, output_login_gallery],
    )

if __name__ == "__main__":
    initialize_system()
    demo.launch(debug=True)
