
import cv2
import numpy as np
import onnxruntime as ort

# ==========================================
# 1. 加载 YOLOv8n ONNX 模型
# ==========================================

MODEL_PATH = "vision/models/yolov8n.onnx"

session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name

# COCO 数据集的 80 个类别
CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

CONF_THRESHOLD = 0.35
IOU_THRESHOLD = 0.45


# ==========================================
# 2. 图片预处理
# ==========================================

def preprocess(image):
    height, width = image.shape[:2]

    # YOLO 输入尺寸
    input_size = 640

    # 等比例缩放，避免图片变形
    scale = min(input_size / width, input_size / height)

    new_width = int(width * scale)
    new_height = int(height * scale)

    resized = cv2.resize(
        image,
        (new_width, new_height)
    )

    # 填充黑边
    padded = np.full(
        (input_size, input_size, 3),
        114,
        dtype=np.uint8
    )

    pad_x = (input_size - new_width) // 2
    pad_y = (input_size - new_height) // 2

    padded[
        pad_y:pad_y + new_height,
        pad_x:pad_x + new_width
    ] = resized

    # BGR 转 RGB
    image_rgb = cv2.cvtColor(
        padded,
        cv2.COLOR_BGR2RGB
    )

    # HWC 转 CHW，并归一化
    input_tensor = image_rgb.transpose(2, 0, 1)
    input_tensor = input_tensor.astype(np.float32) / 255.0

    # 增加 batch 维度
    input_tensor = np.expand_dims(input_tensor, axis=0)

    return input_tensor, scale, pad_x, pad_y


# ==========================================
# 3. 目标检测
# ==========================================

def detect_objects(image):

    input_tensor, scale, pad_x, pad_y = preprocess(image)

    outputs = session.run(
        None,
        {input_name: input_tensor}
    )

    predictions = outputs[0][0].T

    boxes = []
    scores = []
    class_ids = []

    for prediction in predictions:

        class_scores = prediction[4:]

        class_id = int(np.argmax(class_scores))
        confidence = float(class_scores[class_id])

        if confidence < CONF_THRESHOLD:
            continue

        center_x, center_y, width, height = prediction[:4]

        # 转换为左上角坐标
        x1 = center_x - width / 2
        y1 = center_y - height / 2
        x2 = center_x + width / 2
        y2 = center_y + height / 2

        # 去除填充并恢复原始图片坐标
        x1 = (x1 - pad_x) / scale
        y1 = (y1 - pad_y) / scale
        x2 = (x2 - pad_x) / scale
        y2 = (y2 - pad_y) / scale

        image_height, image_width = image.shape[:2]

        x1 = max(0, min(int(x1), image_width - 1))
        y1 = max(0, min(int(y1), image_height - 1))
        x2 = max(0, min(int(x2), image_width - 1))
        y2 = max(0, min(int(y2), image_height - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        boxes.append([x1, y1, x2 - x1, y2 - y1])
        scores.append(confidence)
        class_ids.append(class_id)

    # 非极大值抑制：去除同一物体的重复检测框
    indices = cv2.dnn.NMSBoxes(
        boxes,
        scores,
        CONF_THRESHOLD,
        IOU_THRESHOLD
    )

    results = []

    if len(indices) > 0:
        for index in np.array(indices).flatten():

            x, y, w, h = boxes[index]
            class_id = class_ids[index]
            confidence = scores[index]

            label = CLASSES[class_id]

            results.append({
                "label": label,
                "confidence": confidence,
                "box": [x, y, w, h]
            })

    return results


# ==========================================
# 4. 绘制检测结果
# ==========================================

def draw_detections(image, results):

    for result in results:

        x, y, w, h = result["box"]
        label = result["label"]
        confidence = result["confidence"]

        cv2.rectangle(
            image,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        text = "{} {:.2f}".format(label, confidence)

        cv2.putText(
            image,
            text,
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    return image


# ==========================================
# 5. 使用电脑摄像头测试
# ==========================================

if __name__ == "__main__":

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("无法打开电脑摄像头，请检查摄像头连接。")
        exit()

    print("YOLO 目标检测已启动")
    print("按 Q 键退出")

    while True:

        ret, frame = camera.read()

        if not ret:
            print("无法读取摄像头画面")
            break

        results = detect_objects(frame)

        frame = draw_detections(frame, results)

        cv2.imshow("YOLO Object Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()
