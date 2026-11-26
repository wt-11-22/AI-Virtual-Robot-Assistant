
from pathlib import Path
import onnxruntime as ort
import time
import math
import pybullet as p
import pybullet_data
import cv2
import numpy as np


# ==========================================
# YOLO 模型初始化
# ==========================================

YOLO_MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "vision"
    / "models"
    / "yolov8n.onnx"
)

yolo_session = ort.InferenceSession(
    str(YOLO_MODEL_PATH),
    providers=["CPUExecutionProvider"]
)

yolo_input_name = yolo_session.get_inputs()[0].name

print("YOLO 模型加载成功")


# ==========================================
# 1. 仿真环境
# ==========================================

physics_client = p.connect(p.GUI)

p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setAdditionalSearchPath(r"C:\pybullet_data")

p.setGravity(0, 0, -9.8)
p.setTimeStep(1.0 / 240.0)
p.setRealTimeSimulation(0)

p.loadURDF("plane.urdf")


# ==========================================
# 2. 加载 Laikago
# 保持与 main.py 相同的初始姿态
# ==========================================

robot = p.loadURDF(
    r"C:\pybullet_data\laikago\laikago_toes.urdf",
    basePosition=[0, 0, 0.50],
    baseOrientation=p.getQuaternionFromEuler(
        [1.5707963, 0, 0]
    ),
    useFixedBase=False
)


# ==========================================
# 3. 查看机器人各部分质量
# ==========================================

total_mass = 0.0

for link_index in range(-1, p.getNumJoints(robot)):

    dynamics = p.getDynamicsInfo(robot, link_index)

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
# 4. 四足机器人关节与步态初始化
# ==========================================

leg_groups = {
    "FR": [0, 1, 2],
    "FL": [4, 5, 6],
    "RR": [8, 9, 10],
    "RL": [12, 13, 14]
}

default_angles = [0, 0.7, -1.4]


def set_leg_angles(leg, angles, force=100):

    p.setJointMotorControlArray(
        bodyUniqueId=robot,
        jointIndices=leg_groups[leg],
        controlMode=p.POSITION_CONTROL,
        targetPositions=angles,
        forces=[force] * 3,
        positionGains=[0.2] * 3,
        velocityGains=[1.0] * 3
    )


def stand():

    for leg in leg_groups:
        set_leg_angles(leg, default_angles)


# 初始化关节角度
for leg in leg_groups:

    for index, angle in zip(
        leg_groups[leg],
        default_angles
    ):

        p.resetJointState(
            robot,
            index,
            angle
        )

stand()


# 设置四个脚掌的摩擦力
toe_links = {
    "FR": 3,
    "FL": 7,
    "RR": 11,
    "RL": 15
}

for link in toe_links.values():

    p.changeDynamics(
        robot,
        link,
        lateralFriction=1.5,
        spinningFriction=0.02
    )


# 稳定初始姿态
for _ in range(240):
    p.stepSimulation()


# ==========================================
# 5. 四足机器人步态控制算法
# ==========================================

GAIT_PERIOD = 1.2

HIP_SWING = 0.15

KNEE_SWING = 0.20

leg_phase_offset = {
    "FR": 0.0,
    "RL": 0.0,
    "FL": math.pi,
    "RR": math.pi
}

gait_phase = 0.0


def control_leg(leg, phase, turn):

    if turn > 0:
        side_factor = 1.0 if leg in ["FR", "RR"] else 0.65

    elif turn < 0:
        side_factor = 1.0 if leg in ["FL", "RL"] else 0.65

    else:
        side_factor = 1.0

    leg_phase = phase + leg_phase_offset[leg]

    hip_offset = (
        HIP_SWING *
        side_factor *
        math.sin(leg_phase)
    )

    knee_offset = (
        KNEE_SWING *
        side_factor *
        (1.0 - math.cos(leg_phase)) / 2.0
    )

    target_angles = [
        default_angles[0],
        default_angles[1] + hip_offset,
        default_angles[2] - knee_offset
    ]

    set_leg_angles(
        leg,
        target_angles,
        force=100
    )


def control_gait(phase, turn):

    for leg in leg_groups:
        control_leg(leg, phase, turn)


# ==========================================
# 6. 加载真实盆栽模型
# ==========================================

PLANT_MODEL_DIR = r"C:\robot_assets\plant"

p.setAdditionalSearchPath(PLANT_MODEL_DIR)

plant_model_file = Path(PLANT_MODEL_DIR) / "plant_pybullet.obj"

if not plant_model_file.exists():
    raise FileNotFoundError(
        "找不到盆栽模型: {}".format(plant_model_file)
    )

plant_visual = p.createVisualShape(
    shapeType=p.GEOM_MESH,
    fileName="plant_pybullet.obj",
    meshScale=[0.8, 0.8, 0.8],
    rgbaColor=[1, 1, 1, 1]
)

if plant_visual < 0:
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

    max_confidence = float(np.max(class_scores_all))
    best_index = int(np.argmax(np.max(class_scores_all, axis=1)))

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

    base_position, base_orientation = p.getBasePositionAndOrientation(
        robot
    )

    rotation = p.getMatrixFromQuaternion(base_orientation)

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
# 15. 实时显示与机器人键盘控制
# ==========================================

print("Laikago Camera Started")
print("YOLO Object Detection Started")
print("Press Q to exit.")
print("W: 前进 | S: 后退 | A: 左转 | D: 右转")
print("SPACE: 停止")

detection_results = []

frame_count = 0
DETECTION_INTERVAL = 10

last_time = time.time()

walk_mode = 0
turn_mode = 0


try:

    while p.isConnected():

        # PyBullet 键盘事件读取
        keys = p.getKeyboardEvents()

        def is_down(key):

            return (
                key in keys and
                bool(keys[key] & p.KEY_IS_DOWN)
            )

        w_down = is_down(ord('w'))
        s_down = is_down(ord('s'))
        a_down = is_down(ord('a'))
        d_down = is_down(ord('d'))

        space_down = is_down(
            getattr(p, "B3G_SPACE", 32)
        )

        # 根据按键确定运动模式
        if space_down:

            walk_mode = 0
            turn_mode = 0

        else:

            if w_down:
                walk_mode = 1
            elif s_down:
                walk_mode = -1
            else:
                walk_mode = 0

            if a_down:
                turn_mode = 1
            elif d_down:
                turn_mode = -1
            else:
                turn_mode = 0

        # 更新步态相位
        now = time.time()

        dt = min(now - last_time, 0.02)

        last_time = now

        if walk_mode != 0 or turn_mode != 0:

            direction = walk_mode

            if direction == 0:
                direction = 1

            gait_phase += (
                direction *
                2.0 *
                math.pi *
                dt /
                GAIT_PERIOD
            )

            gait_phase %= 2.0 * math.pi

            control_gait(
                gait_phase,
                turn_mode
            )

        else:

            stand()

        # 执行物理仿真
        p.stepSimulation()

        # 获取 RGB 和深度图
        rgb_image, depth_image = capture_camera()

        # 保存第一帧摄像头图片
        if frame_count == 0:

            cv2.imwrite(
                "camera_test.jpg",
                rgb_image
            )

            print("摄像头图片已保存：camera_test.jpg")

        frame_count += 1

        # 定期执行 YOLO 检测
        if frame_count % DETECTION_INTERVAL == 0:

            detection_results = yolo_detect(rgb_image)

            if detection_results:

                print(
                    "检测到 {} 个物体".format(
                        len(detection_results)
                    )
                )

                for result in detection_results:

                    class_name = YOLO_CLASSES[
                        result["class_id"]
                    ]

                    confidence = result["confidence"]

                    distance = estimate_object_distance(
                        depth_image,
                        result["box"]
                    )

                    if distance is not None:

                        print(
                            "物体：{} | 置信度：{:.2f} | 距离：{:.2f} 米".format(
                                class_name,
                                confidence,
                                distance
                            )
                        )

                    else:

                        print(
                            "物体：{} | 无法获取有效距离".format(
                                class_name
                            )
                        )

        # 绘制检测框
        display_image = yolo_draw(
            rgb_image.copy(),
            detection_results
        )

        cv2.imshow(
            "Robot Dog - YOLO Detection",
            display_image
        )

        # 深度图显示
        depth_display = np.clip(
            depth_image / FAR * 255,
            0,
            255
        ).astype(np.uint8)

        depth_display = cv2.applyColorMap(
            depth_display,
            cv2.COLORMAP_JET
        )

        cv2.imshow(
            "Robot Dog - Depth Camera",
            depth_display
        )

        # 退出检测
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        time.sleep(1.0 / 240.0)


finally:

    cv2.destroyAllWindows()

    if p.isConnected():
        p.disconnect()

    print("Camera closed.")
