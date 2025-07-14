# file: face_detector.py (phiên bản ONNX)
import os
import numpy as np
import onnxruntime as ort
from skimage import transform
from skimage.io import imread
import cv2

DEFAULT_ANCHORS = np.array(
    [[[0.5, 0.625], [1.0, 1.25], [1.625, 2.0]],
     [[1.4375, 1.8125], [2.6875, 3.4375], [4.5625, 6.5625]],
     [[4.5625, 6.781199932098389], [7.218800067901611, 9.375], [10.468999862670898, 13.531000137329102]]],
    dtype="float32"
)
DEFAULT_STRIDES = np.array([8, 16, 32], dtype="float32")

class YoloV5FaceDetector:
    def __init__(self, model_path, anchors=DEFAULT_ANCHORS, strides=DEFAULT_STRIDES):
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        print(f"YOLOv5 Detector ONNX đang chạy trên: {self.session.get_providers()}")
        self.anchors, self.strides = anchors, strides
        self.num_anchors = anchors.shape[1]
        self.anchor_grids = np.ceil((anchors * strides[:, np.newaxis, np.newaxis])[:, np.newaxis, :, np.newaxis, :])

    def _face_align_landmarks(self, img, landmarks, image_size=(112, 112)):
        tform = transform.SimilarityTransform()
        src = np.array(
            [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
             [41.5493, 92.3655], [70.729904, 92.2041]], dtype=np.float32)
        ret = []
        for landmark in landmarks:
            tform.estimate(landmark, src)
            ndimage = transform.warp(img, tform.inverse, output_shape=image_size)
            if len(ndimage.shape) == 2:
                ndimage = np.stack([ndimage, ndimage, ndimage], -1)
            ret.append(ndimage)
        return (np.array(ret) * 255).astype(np.uint8)

    def _make_grid(self, nx, ny):
        xv, yv = np.meshgrid(np.arange(nx), np.arange(ny))
        return np.stack((xv, yv), 2).reshape((1, 1, -1, 2)).astype(np.float32)

    def _post_process(self, outputs, image_height, image_width):
        post_outputs = []
        for output, stride, anchor, anchor_grid in zip(outputs, self.strides, self.anchors, self.anchor_grids):
            hh, ww = int(np.ceil(image_height / stride)), int(np.ceil(image_width / stride))
            anchor_width = output.shape[-1] // self.num_anchors
            output = output.reshape(-1, output.shape[1] * output.shape[2], self.num_anchors, anchor_width).transpose(0, 2, 1, 3)

            def sigmoid(x): return 1 / (1 + np.exp(-x))
            cls = sigmoid(output[:, :, :, :5])
            cur_grid = self._make_grid(ww, hh) * stride
            xy = cls[:, :, :, 0:2] * (2 * stride) - 0.5 * stride + cur_grid
            wh = (cls[:, :, :, 2:4] * 2)**2 * anchor_grid
            
            mm = [1, 1, 1, 5]
            landmarks = output[:, :, :, 5:15] * np.tile(anchor_grid, mm) + np.tile(cur_grid, mm)
            
            post_out = np.concatenate([xy, wh, landmarks, cls[:, :, :, 4:]], axis=-1)
            post_outputs.append(post_out.reshape(-1, output.shape[1] * output.shape[2], anchor_width - 1))
        return np.concatenate(post_outputs, axis=1)

    def _yolo_nms(self, inputs, max_output_size=15, iou_threshold=0.45, score_threshold=0.25):
        inputs = inputs[0][inputs[0, :, -1] > score_threshold]
        xy_center, wh, ppt, cct = inputs[:, :2], inputs[:, 2:4], inputs[:, 4:14], inputs[:, 14]
        xy_start = xy_center - wh / 2
        boxes = np.concatenate([xy_start, wh], axis=-1)
        
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), cct.tolist(), score_threshold, iou_threshold)
        
        if len(indices) == 0:
            return np.array([]), np.array([]), np.array([])
            
        bbs = np.concatenate([xy_start[indices], xy_start[indices] + wh[indices]], axis=-1)
        pps = np.reshape(ppt[indices], [-1, 5, 2])
        ccs = cct[indices]
        return bbs, pps, ccs

    def detect_in_image(self, image, max_output_size=15, iou_threshold=0.45, score_threshold=0.25):
        if isinstance(image, str):
            image = imread(image)
        
        image_rgb = image[:, :, ::-1] if len(image.shape) > 2 and image.shape[2] == 3 else image
        
        hh, ww, _ = image_rgb.shape
        pad_hh = (32 - hh % 32) % 32
        pad_ww = (32 - ww % 32) % 32
        if pad_ww != 0 or pad_hh != 0:
            image_rgb = np.pad(image_rgb, [[0, pad_hh], [0, pad_ww], [0, 0]], 'constant')

        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: np.expand_dims(image_rgb, 0).astype('uint8')})
        
        post_outputs = self._post_process(outputs, image_rgb.shape[0], image_rgb.shape[1])
        bbs, pps, ccs = self._yolo_nms(post_outputs, max_output_size, iou_threshold, score_threshold)
        
        aligned_faces = np.array([])
        if len(bbs) != 0:
            aligned_faces = self._face_align_landmarks(image_rgb, pps)
            
        return bbs, pps, ccs, aligned_faces