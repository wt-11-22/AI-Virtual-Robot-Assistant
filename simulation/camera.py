
from pathlib import Path
import sys
import time
import threading
import queue

import onnxruntime as ort
import pybullet as p
import cv2
import numpy as np


# ==========================================
# 1. 添加项目根目录
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ==========================================
# 2. 导入统一控制接口
# ==========================================

from robot.robot_controller import RobotController
from robot.keyboard_controller import KeyboardController


# ==========================================
# 3. YOLO 模型初始化
# ==========================================

YOLO_MODEL_PATH = (
    PROJECT_ROOT
    / "vision"
    / "models"
    / "yolov8n.onnx"
)

if not YOLO_MODEL_PATH.exists():
    raise FileNotFoundError(
        "找不到 YOLO 模型: {}".format(YOLO_MODEL_PATH)
    )

yolo_session = ort.InferenceSession(
    str(YOLO_MODEL_PATH),
    providers=["CPUExecutionProvider"]
)

yolo_input_name = yolo_session.get_inputs()[0].name

print("YOLO 模型加载成功")


# ==========================================
# 4. 启动机器人和键盘控制器
# ==========================================

robot = RobotController()
keyboard_controller = KeyboardController()

robot.start()

robot_id = robot.robot_id

print("Laikago 机器人启动成功")


# ==========================================
# 5. 查看机器人各部分质量
# ==========================================

total_mass = 0.0

for link_index in range(-1, p.getNumJoints(robot_id)):

    dynamics = p.getDynamicsInfo(robot_id, link_index)

    mass = dynamics[0]
    total_mass += mass

    print(
        "Link:", link_index,
        "质量:", mass, "kg"
    )

print("=" * 40)
print("机器人总质量:", total_mass, "kg")
print("=" * 40)


# ==========================================
# 6. 加载真实盆栽模型
# ==========================================

PLANT_MODEL_DIR = Path(r"C:\robot_assets\plant")

plant_model_file = (
    PLANT_MODEL_DIR / "plant_pybullet.obj"
)

if not plant_model_file.exists():
    robot.close()
    raise FileNotFoundError(
        "找不到盆栽模型: {}".format(plant_model_file)
    )

plant_visual = p.createVisualShape(
    shapeType=p.GEOM_MESH,
    fileName=str(plant_model_file),
    meshScale=[0.8, 0.8, 0.8],
    rgbaColor=[1, 1, 1, 1]
)

if plant_visual < 0:
    robot.close()
    raise RuntimeError("盆栽 OBJ 模型加载失败！")

plant = p.createMultiBody(
    baseMass=0,
    baseVisualShapeIndex=plant_visual,
    basePosition=[0, -2.0, 0]
)

print("盆栽模型加载成功，模型ID:", plant)


# ==========================================
# 7. YOLO 类别
# ==========================================

YOLO_CLASSES = [
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

YOLO_CONF_THRESHOLD = 0.35
YOLO_IOU_THRESHOLD = 0.45


# ==========================================
# 8. YOLO 图片预处理
# ==========================================

def yolo_preprocess(image):

    height, width = image.shape[:2]
    input_size = 640

    scale = min(
        input_size / width,
        input_size / height
    )

    new_width = int(round(width * scale))
    new_height = int(round(height * scale))

    resized = cv2.resize(
        image,
        (new_width, new_height)
    )

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

    rgb = cv2.cvtColor(
        padded,
        cv2.COLOR_BGR2RGB
    )

    tensor = rgb.transpose(2, 0, 1)
    tensor = tensor.astype(np.float32) / 255.0
    tensor = np.expand_dims(tensor, axis=0)

    return tensor, scale, pad_x, pad_y


# ==========================================
# 9. YOLO 目标检测
# ==========================================

def yolo_detect(image):

    tensor, scale, pad_x, pad_y = yolo_preprocess(image)

    outputs = yolo_session.run(
        None,
        {yolo_input_name: tensor}
    )

    predictions = outputs[0][0].T

    class_scores_all = predictions[:, 4:]

    max_confidence = float(
        np.max(class_scores_all)
    )

    best_index = int(
        np.argmax(
            np.max(class_scores_all, axis=1)
        )
    )

    best_class_id = int(
        np.argmax(class_scores_all[best_index])
    )

    print(
        "YOLO最高置信度：{:.4f}，类别：{}".format(
            max_confidence,
            YOLO_CLASSES[best_class_id]
        )
    )

    boxes = []
    scores = []
    class_ids = []

    image_height, image_width = image.shape[:2]

    for prediction in predictions:

        class_scores = prediction[4:]

        class_id = int(np.argmax(class_scores))
        confidence = float(class_scores[class_id])

        if confidence < YOLO_CONF_THRESHOLD:
            continue

        center_x, center_y, box_width, box_height = prediction[:4]

        x1 = center_x - box_width / 2
        y1 = center_y - box_height / 2
        x2 = center_x + box_width / 2
        y2 = center_y + box_height / 2

        x1 = (x1 - pad_x) / scale
        y1 = (y1 - pad_y) / scale
        x2 = (x2 - pad_x) / scale
        y2 = (y2 - pad_y) / scale

        x1 = max(0, min(int(x1), image_width - 1))
        y1 = max(0, min(int(y1), image_height - 1))
        x2 = max(0, min(int(x2), image_width - 1))
        y2 = max(0, min(int(y2), image_height - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        boxes.append([
            x1,
            y1,
            x2 - x1,
            y2 - y1
        ])

        scores.append(confidence)
        class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(
        boxes,
        scores,
        YOLO_CONF_THRESHOLD,
        YOLO_IOU_THRESHOLD
    )

    results = []

    if len(indices) > 0:

        for index in np.array(indices).flatten():

            results.append({
                "box": boxes[index],
                "class_id": class_ids[index],
                "confidence": scores[index]
            })

    return results


# ==========================================
# 10. 绘制目标检测结果
# ==========================================

def yolo_draw(image, results):

    for result in results:

        x, y, width, height = result["box"]

        class_id = result["class_id"]
        confidence = result["confidence"]

        label = YOLO_CLASSES[class_id]

        cv2.rectangle(
            image,
            (x, y),
            (x + width, y + height),
            (0, 255, 0),
            2
        )

        text = "{} {:.2f}".format(
            label,
            confidence
        )

        cv2.putText(
            image,
            text,
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2
        )

    return image


# ==========================================
# 11. 摄像头参数
# ==========================================

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

FOV = 70
NEAR = 0.05
FAR = 10.0

projection_matrix = p.computeProjectionMatrixFOV(
    fov=FOV,
    aspect=IMAGE_WIDTH / IMAGE_HEIGHT,
    nearVal=NEAR,
    farVal=FAR
)

CAMERA_OFFSET = [0.0, 0.0, 0.25]

CAMERA_FORWARD = [0.0, 0.0, 1.0]

CAMERA_UP = [0.0, 1.0, 0.0]


# ==========================================
# 12. 根据机身姿态计算摄像头位置
# ==========================================

def get_camera_matrices():

    base_position, base_orientation = (
        p.getBasePositionAndOrientation(robot_id)
    )

    rotation = p.getMatrixFromQuaternion(
        base_orientation
    )

    rotation = np.array(rotation).reshape(3, 3)

    offset = rotation @ np.array(CAMERA_OFFSET)

    camera_position = (
        np.array(base_position) + offset
    )

    forward = rotation @ np.array(CAMERA_FORWARD)
    up = rotation @ np.array(CAMERA_UP)

    camera_target = camera_position + forward

    view_matrix = p.computeViewMatrix(
        cameraEyePosition=camera_position.tolist(),
        cameraTargetPosition=camera_target.tolist(),
        cameraUpVector=up.tolist()
    )

    return view_matrix


# ==========================================
# 13. RGB 和深度摄像头
# ==========================================

def capture_camera():

    view_matrix = get_camera_matrices()

    image = p.getCameraImage(
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix,
        renderer=p.ER_TINY_RENDERER
    )

    rgba = np.asarray(
        image[2],
        dtype=np.uint8
    ).reshape(
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
        4
    )

    rgb = cv2.cvtColor(
        rgba,
        cv2.COLOR_RGBA2BGR
    )

    depth_buffer = np.asarray(
        image[3],
        dtype=np.float32
    ).reshape(
        IMAGE_HEIGHT,
        IMAGE_WIDTH
    )

    depth = FAR * NEAR / (
        FAR - (FAR - NEAR) * depth_buffer
    )

    return rgb, depth


# ==========================================
# 14. 物体距离估算
# ==========================================

def estimate_object_distance(depth_image, box):

    x, y, width, height = box

    roi = depth_image[
        y:y + height,
        x:x + width
    ]

    valid_depths = roi[
        (roi > NEAR) &
        (roi < FAR * 0.98)
    ]

    if len(valid_depths) == 0:
        return None

    distance = float(np.median(valid_depths))

    return distance


# ==========================================
# 15. 视觉线程：YOLO 后台检测
# ==========================================

# 队列最多保留一张待检测图片。
# 如果旧图片还没有处理完，就用最新图片替换。
yolo_queue = queue.Queue(maxsize=1)

yolo_stop_event = threading.Event()

# 保存最新检测结果及其对应的深度图
yolo_result_lock = threading.Lock()

shared_detection_data = {
    "results": [],
    "depth": None,
    "timestamp": 0.0,
    "version": 0
}


def yolo_worker():

    print("YOLO 后台检测线程启动")

    while not yolo_stop_event.is_set():

        try:
            image, depth_image = yolo_queue.get(timeout=0.1)

        except queue.Empty:
            continue

        try:

            # YOLO 推理只在后台线程执行
            results = yolo_detect(image)

            # 计算每个目标的距离
            for result in results:

                distance = estimate_object_distance(
                    depth_image,
                    result["box"]
                )

                result["distance"] = distance

            # 更新共享检测结果
            with yolo_result_lock:

                shared_detection_data["results"] = results
                shared_detection_data["depth"] = depth_image
                shared_detection_data["timestamp"] = time.time()
                shared_detection_data["version"] += 1

            if len(results) == 0:

                print("YOLO：当前没有检测到目标")

            else:

                for result in results:

                    class_name = YOLO_CLASSES[
                        result["class_id"]
                    ]

                    distance = result.get("distance")

                    if distance is None:
                        distance_text = "未知"

                    else:
                        distance_text = "{:.2f} 米".format(
                            distance
                        )

                    print(
                        "检测目标：{} | 置信度：{:.2f} | 距离：{}".format(
                            class_name,
                            result["confidence"],
                            distance_text
                        )
                    )

        except Exception as error:

            print("YOLO 检测线程发生错误:", error)

        finally:
            yolo_queue.task_done()

    print("YOLO 后台检测线程退出")


# ==========================================
# 16. 启动 YOLO 后台线程
# ==========================================

yolo_thread = threading.Thread(
    target=yolo_worker,
    name="YOLO-Worker",
    daemon=True
)

yolo_thread.start()


# ==========================================
# 17. 运行参数
# ==========================================

# 每隔多少次机器人控制循环采集一次摄像头
CAMERA_INTERVAL = 4

# 两次 YOLO 图片提交之间至少间隔多少秒
DETECTION_INTERVAL_SECONDS = 0.5

frame_count = 0

last_detection_submit_time = 0.0

display_results = []

last_result_version = 0

last_rgb = None
last_depth = None


# ==========================================
# 18. 键盘控制说明
# ==========================================

print("\n========== AI Virtual Robot Assistant ==========")
print("W：前进")
print("S：后退")
print("A：左转")
print("D：右转")
print("Q：原地左旋转")
print("E：原地右旋转")
print("SPACE：停止")
print("ESC：退出")
print("-----------------------------------------------")
print("RGB 摄像头：已启用")
print("深度摄像头：已启用")
print("YOLO 目标检测：后台线程运行")
print("注意：操作时让 PyBullet 窗口处于焦点状态")
print("================================================")


# ==========================================
# 19. 主循环：机器人运动优先
# ==========================================

try:

    while p.isConnected():

        # ----------------------------------
        # A. 读取键盘
        # ----------------------------------

        command = keyboard_controller.get_command()

        if command["exit"]:
            break

        if command["stop"]:

            robot.stop()

        else:

            robot.set_motion(
                walk=command["walk"],
                turn=command["turn"],
                spin=command["spin"]
            )

        # ----------------------------------
        # B. 优先更新机器人
        # ----------------------------------

        robot.update()

        frame_count += 1

        # ----------------------------------
        # C. 定期采集摄像头
        # ----------------------------------

        if frame_count % CAMERA_INTERVAL == 0:

            rgb_image, depth_image = capture_camera()

            last_rgb = rgb_image
            last_depth = depth_image

            # ----------------------------------
            # D. 定期向 YOLO 后台线程提交图片
            # ----------------------------------

            current_time = time.time()

            if (
                current_time - last_detection_submit_time
                >= DETECTION_INTERVAL_SECONDS
            ):

                image_copy = rgb_image.copy()
                depth_copy = depth_image.copy()

                try:

                    # 尝试直接放入队列
                    yolo_queue.put_nowait(
                        (image_copy, depth_copy)
                    )

                    last_detection_submit_time = current_time

                except queue.Full:

                    # 如果旧图片还没处理完，丢弃旧图片
                    # 优先保留最新摄像头画面
                    try:
                        yolo_queue.get_nowait()
                        yolo_queue.task_done()

                    except queue.Empty:
                        pass

                    try:

                        yolo_queue.put_nowait(
                            (image_copy, depth_copy)
                        )

                        last_detection_submit_time = current_time

                    except queue.Full:
                        pass

        # ----------------------------------
        # E. 读取后台线程的最新检测结果
        # ----------------------------------

        with yolo_result_lock:

            if (
                shared_detection_data["version"]
                != last_result_version
            ):

                display_results = [
                    dict(item)
                    for item in shared_detection_data["results"]
                ]

                last_result_version = (
                    shared_detection_data["version"]
                )

        # ----------------------------------
        # F. 显示 RGB 和 YOLO 检测窗口
        # ----------------------------------

        if last_rgb is not None:

            display_image = last_rgb.copy()

            display_image = yolo_draw(
                display_image,
                display_results
            )

            cv2.imshow(
                "Robot RGB + YOLO Detection",
                display_image
            )

        # ----------------------------------
        # G. 显示深度图
        # ----------------------------------

        if last_depth is not None:

            depth_display = np.clip(
                last_depth / FAR * 255,
                0,
                255
            ).astype(np.uint8)

            depth_display = cv2.applyColorMap(
                depth_display,
                cv2.COLORMAP_JET
            )

            cv2.imshow(
                "Robot Depth Camera",
                depth_display
            )

        # ----------------------------------
        # H. 处理 OpenCV 窗口事件
        # ----------------------------------

        cv2.waitKey(1)

        # ----------------------------------
        # I. 控制循环休眠
        # ----------------------------------

        time.sleep(1.0 / 240.0)


except KeyboardInterrupt:

    print("\n用户中断仿真。")


except Exception as error:

    print("\n主程序发生错误:", error)


finally:

    # ======================================
    # 20. 安全关闭程序
    # ======================================

    print("\n正在关闭视觉系统...")

    yolo_stop_event.set()

    if yolo_thread.is_alive():

        yolo_thread.join(timeout=2.0)

    cv2.destroyAllWindows()

    robot.close()

    print("机器人仿真已关闭。")
    print("视觉系统已关闭。")
