
import time
import math
import pybullet as p

# ==========================================
# 1. 启动仿真环境
# ==========================================

physics_client = p.connect(p.GUI)

p.setAdditionalSearchPath(r"C:\pybullet_data")
p.setGravity(0, 0, -9.8)
p.setTimeStep(1.0 / 240.0)
p.setRealTimeSimulation(0)

plane_id = p.loadURDF("plane.urdf")

p.changeDynamics(
    plane_id,
    -1,
    lateralFriction=1.5,
    spinningFriction=0.02
)

# ==========================================
# 2. 加载 Laikago
# ==========================================

orientation = p.getQuaternionFromEuler(
    [1.5707963, 0, 0]
)

robot_id = p.loadURDF(
    r"C:\pybullet_data\laikago\laikago_toes.urdf",
    basePosition=[0, 0, 0.50],
    baseOrientation=orientation,
    useFixedBase=False
)

p.resetDebugVisualizerCamera(
    cameraDistance=2.5,
    cameraYaw=45,
    cameraPitch=-20,
    cameraTargetPosition=[0, 0, 0.3]
)

# ==========================================
# 3. 识别四条腿
# ==========================================

leg_groups = {
    "FR": [],
    "FL": [],
    "RR": [],
    "RL": []
}

for i in range(p.getNumJoints(robot_id)):
    info = p.getJointInfo(robot_id, i)
    name = info[1].decode("utf-8")

    if info[2] == p.JOINT_REVOLUTE:
        for leg in leg_groups:
            if name.startswith(leg):
                leg_groups[leg].append(i)

print("\n========== 四条腿识别结果 ==========")

for leg, joints in leg_groups.items():
    print(leg, joints)

    if len(joints) != 3:
        raise RuntimeError(
            f"{leg} 腿没有识别到三个旋转关节"
        )

# 按照 FR、FL、RR、RL 顺序排列
joint_indices = []

for leg in ["FR", "FL", "RR", "RL"]:
    joint_indices.extend(leg_groups[leg])

if len(joint_indices) != 12:
    raise RuntimeError("关节数量不符合预期")

# ==========================================
# 4. 初始站立姿态
# ==========================================

default_angles = [0, 0.7, -1.4]

def set_leg_angles(leg, angles, force=100):

    p.setJointMotorControlArray(
        bodyUniqueId=robot_id,
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

# 初始化关节
for leg in leg_groups:
    for index, angle in zip(
        leg_groups[leg],
        default_angles
    ):
        p.resetJointState(robot_id, index, angle)

stand()

# ==========================================
# 5. 脚掌摩擦力
# ==========================================

toe_links = {
    "FR": 3,
    "FL": 7,
    "RR": 11,
    "RL": 15
}

for link in toe_links.values():

    p.changeDynamics(
        robot_id,
        link,
        lateralFriction=1.5,
        spinningFriction=0.02
    )

# ==========================================
# 6. 初始姿态稳定
# ==========================================

print("\n正在稳定初始姿态...")

for _ in range(240):
    p.stepSimulation()
    time.sleep(1.0 / 240.0)

print("初始姿态稳定完成")

# ==========================================
# 7. 步态参数
# ==========================================

# 步态周期，数值越大，运动越慢
GAIT_PERIOD = 1.2

# 髋关节摆动幅度
HIP_SWING = 0.15

# 膝关节摆动幅度
KNEE_SWING = 0.20

# 对角腿采用相反相位
leg_phase_offset = {
    "FR": 0.0,
    "RL": 0.0,
    "FL": math.pi,
    "RR": math.pi
}

# 步态相位
gait_phase = 0.0

# ==========================================
# 8. 单腿关节控制
# ==========================================

def control_leg(leg, phase, turn):

    # 根据左右转向调整摆动幅度
    if turn > 0:
        side_factor = (
            1.0 if leg in ["FR", "RR"] else 0.65
        )

    elif turn < 0:
        side_factor = (
            1.0 if leg in ["FL", "RL"] else 0.65
        )

    else:
        side_factor = 1.0

    # 计算当前腿的相位
    leg_phase = (
        phase + leg_phase_offset[leg]
    )

    # 使用正弦曲线生成连续关节运动
    hip_offset = (
        HIP_SWING
        * side_factor
        * math.sin(leg_phase)
    )

    knee_offset = (
        KNEE_SWING
        * side_factor
        * (1.0 - math.cos(leg_phase))
        / 2.0
    )

    # 以原始站立角度为中心
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

# ==========================================
# 9. 四足步态控制
# ==========================================

def control_gait(phase, turn):

    for leg in leg_groups:

        control_leg(
            leg,
            phase,
            turn
        )

# ==========================================
# 10. 键盘控制说明
# ==========================================

print("\n========== Laikago 步态控制 ==========")
print("W：前进")
print("S：后退")
print("A：左转")
print("D：右转")
print("W + A：向前左转")
print("W + D：向前右转")
print("S + A：向后左转")
print("S + D：向后右转")
print("空格：停止并站立")
print("======================================")

# ==========================================
# 11. 主循环
# ==========================================

last_time = time.time()
last_print_time = time.time()

walk_mode = 0
turn_mode = 0

try:

    while True:

        keys = p.getKeyboardEvents()

        def is_down(key):

            return (
                key in keys
                and bool(keys[key] & p.KEY_IS_DOWN)
            )

        # 读取键盘
        w_down = is_down(ord("w"))
        s_down = is_down(ord("s"))
        a_down = is_down(ord("a"))
        d_down = is_down(ord("d"))

        # PyBullet 空格键
        space_key = getattr(p, "B3G_SPACE", 32)
        space_down = is_down(space_key)

        # ==================================
        # 空格键最高优先级
        # ==================================

        if space_down:

            walk_mode = 0
            turn_mode = 0

        else:

            # 前进和后退
            if w_down:
                walk_mode = 1

            elif s_down:
                walk_mode = -1

            else:
                walk_mode = 0

            # 左右转向
            if a_down:
                turn_mode = 1

            elif d_down:
                turn_mode = -1

            else:
                turn_mode = 0

        # ==================================
        # 计算时间间隔
        # ==================================

        now = time.time()

        dt = min(
            now - last_time,
            0.02
        )

        last_time = now

        # ==================================
        # 执行步态
        # ==================================

        if walk_mode != 0 or turn_mode != 0:

            # 后退时反向推进步态相位
            direction = walk_mode

            if direction == 0:
                direction = 1

            gait_phase += (
                direction
                * 2.0
                * math.pi
                * dt
                / GAIT_PERIOD
            )

            gait_phase %= 2.0 * math.pi

            control_gait(
                gait_phase,
                turn_mode
            )

        else:

            # 停止后恢复站立姿态
            stand()

        # ==================================
        # 物理仿真
        # ==================================

        p.stepSimulation()

        # ==================================
        # 每两秒打印机身位置
        # ==================================

        if now - last_print_time >= 2.0:

            position, _ = (
                p.getBasePositionAndOrientation(robot_id)
            )

            print(
                "机身位置:",
                [round(x, 3) for x in position],
                "运动模式:",
                walk_mode,
                "转向模式:",
                turn_mode
            )

            last_print_time = now

        time.sleep(1.0 / 240.0)

except KeyboardInterrupt:

    print("\n正在关闭仿真...")

finally:

    p.disconnect()
