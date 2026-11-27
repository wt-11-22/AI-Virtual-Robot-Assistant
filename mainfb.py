import time
import math
import pybullet as p

# ==========================================
# 1. 启动仿真环境
# ==========================================

physics_client = p.connect(p.GUI)

if physics_client < 0:
    raise RuntimeError("PyBullet 仿真环境启动失败")

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

orientation = p.getQuaternionFromEuler([1.5707963, 0, 0])

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
# 3. 识别四条腿和关节
# ==========================================

leg_groups = {
    "FR": [],
    "FL": [],
    "RR": [],
    "RL": []
}

for joint_index in range(p.getNumJoints(robot_id)):
    info = p.getJointInfo(robot_id, joint_index)
    joint_name = info[1].decode("utf-8")

    if info[2] == p.JOINT_REVOLUTE:
        for leg in leg_groups:
            if joint_name.startswith(leg):
                leg_groups[leg].append(joint_index)

print("\n========== 四条腿识别结果 ==========")

for leg, joints in leg_groups.items():
    print(leg, joints)

    if len(joints) != 3:
        raise RuntimeError(
            f"{leg} 腿应有3个旋转关节，实际识别到 {len(joints)} 个"
        )

all_leg_joints = [
    joint
    for leg in ["FR", "FL", "RR", "RL"]
    for joint in leg_groups[leg]
]

if len(set(all_leg_joints)) != 12:
    raise RuntimeError("四条腿关节重复或数量不正确")

# PyBullet IK 返回值按照所有非固定关节的顺序排列
movable_joint_indices = []

for joint_index in range(p.getNumJoints(robot_id)):
    info = p.getJointInfo(robot_id, joint_index)

    if info[2] != p.JOINT_FIXED:
        movable_joint_indices.append(joint_index)

if len(movable_joint_indices) != 12:
    raise RuntimeError(
        f"当前代码按12个可运动关节配置，实际识别到 "
        f"{len(movable_joint_indices)} 个"
    )

ik_joint_map = {
    joint: index
    for index, joint in enumerate(movable_joint_indices)
}

# ==========================================
# 4. 初始站立姿态和关节控制
# ==========================================

default_angles = [0.0, 0.7, -1.4]


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
    for leg in ["FR", "FL", "RR", "RL"]:
        set_leg_angles(leg, default_angles)


for leg in ["FR", "FL", "RR", "RL"]:
    for joint_index, angle in zip(
        leg_groups[leg],
        default_angles
    ):
        p.resetJointState(robot_id, joint_index, angle)

stand()

# ==========================================
# 5. 脚掌摩擦力和脚掌链接
# ==========================================

toe_links = {
    "FR": 3,
    "FL": 7,
    "RR": 11,
    "RL": 15
}

num_joints = p.getNumJoints(robot_id)

for leg, link_index in toe_links.items():
    if link_index < 0 or link_index >= num_joints:
        raise RuntimeError(
            f"{leg} 脚掌链接编号 {link_index} 超出模型范围"
        )

    p.changeDynamics(
        robot_id,
        link_index,
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
# 7. 获取四只脚掌的初始位置
# ==========================================

base_position, base_orientation = (
    p.getBasePositionAndOrientation(robot_id)
)

inverse_base_position, inverse_base_orientation = (
    p.invertTransform(base_position, base_orientation)
)

nominal_foot_positions = {}

for leg, link_index in toe_links.items():
    link_state = p.getLinkState(
        robot_id,
        link_index,
        computeForwardKinematics=True
    )

    # linkState[4] 是世界坐标系中的 worldLinkFramePosition
    foot_world_position = link_state[4]

    foot_local_position, _ = p.multiplyTransforms(
        inverse_base_position,
        inverse_base_orientation,
        foot_world_position,
        [0, 0, 0, 1]
    )

    nominal_foot_positions[leg] = list(foot_local_position)

    print(
        leg,
        "初始脚掌位置:",
        [round(value, 4) for value in foot_local_position]
    )

# ==========================================
# 8. 自动计算机身前进方向
# ==========================================

front_center = [
    (
        nominal_foot_positions["FR"][axis]
        + nominal_foot_positions["FL"][axis]
    ) / 2.0
    for axis in range(3)
]

rear_center = [
    (
        nominal_foot_positions["RR"][axis]
        + nominal_foot_positions["RL"][axis]
    ) / 2.0
    for axis in range(3)
]

forward_vector = [
    front_center[axis] - rear_center[axis]
    for axis in range(3)
]

forward_length = math.sqrt(
    sum(value * value for value in forward_vector)
)

if forward_length < 0.001:
    raise RuntimeError(
        "无法确定机身前进方向，请检查 FR/FL/RR/RL 腿部识别结果"
    )

forward_vector = [
    value / forward_length
    for value in forward_vector
]

print(
    "\n机身坐标系前进方向:",
    [round(value, 4) for value in forward_vector]
)

# 计算机身左侧方向（与前进方向正交）
left_center = [
    (
        nominal_foot_positions["FL"][axis]
        + nominal_foot_positions["RL"][axis]
    ) / 2.0
    for axis in range(3)
]

right_center = [
    (
        nominal_foot_positions["FR"][axis]
        + nominal_foot_positions["RR"][axis]
    ) / 2.0
    for axis in range(3)
]

lateral_vector = [
    left_center[i] - right_center[i]
    for i in range(3)
]

projection = sum(
    lateral_vector[i] * forward_vector[i]
    for i in range(3)
)

lateral_vector = [
    lateral_vector[i] - projection * forward_vector[i]
    for i in range(3)
]

lateral_length = math.sqrt(
    sum(value * value for value in lateral_vector)
)

if lateral_length < 0.001:
    raise RuntimeError("无法确定机身左右方向")

lateral_vector = [
    value / lateral_length
    for value in lateral_vector
]

# ==========================================
# 9. 步态与转向参数
# ==========================================

GAIT_PERIOD = 1.6
SWING_RATIO = 0.25
STEP_LENGTH = 0.08
STEP_HEIGHT = 0.035

# 原来的行进转向参数，保持不变
TURN_GAIN = 0.35

# Q/E 原地旋转参数
SPIN_PERIOD = 1.8
SPIN_STEP_LENGTH = 0.055
SPIN_STEP_HEIGHT = 0.025

# 四拍顺序：FR -> RL -> FL -> RR
leg_phase_offset = {
    "FR": 0.00,
    "RL": 0.25,
    "FL": 0.50,
    "RR": 0.75
}

gait_phase = 0.0

RAMP_TIME = 1.0
CONTROL_DT = 1.0 / 60.0

# ==========================================
# 10. 平滑插值
# ==========================================

def smooth_step(value):
    value = max(0.0, min(1.0, value))
    return 3.0 * value * value - 2.0 * value * value * value

# ==========================================
# 11. 获取当前IK的关节参考角度
# ==========================================

def get_ik_rest_poses():
    """
    获取当前所有可运动关节角度，作为IK参考姿态。
    """

    rest_poses = []

    for joint_index in movable_joint_indices:
        joint_state = p.getJointState(
            robot_id,
            joint_index
        )

        rest_poses.append(joint_state[0])

    return rest_poses

# ==========================================
# 12. 计算单腿脚掌目标位置
# ==========================================

def calculate_foot_target(leg, phase, stride_scale, turn, spin=0):

    # 左转：右侧为外侧腿，左侧为内侧腿
    if turn > 0:
        stride_factor = (
            1.0 + TURN_GAIN
            if leg in ["FR", "RR"]
            else 1.0 - TURN_GAIN
        )

    # 右转：左侧为外侧腿，右侧为内侧腿
    elif turn < 0:
        stride_factor = (
            1.0 + TURN_GAIN
            if leg in ["FL", "RL"]
            else 1.0 - TURN_GAIN
        )

    else:
        stride_factor = 1.0

    # 当前腿的独立相位
    leg_phase = (
        phase - leg_phase_offset[leg]
    ) % 1.0

    # 普通行走参数保持原样
    # Q/E 使用独立的旋转步幅
    if spin == 0:
        stride = STEP_LENGTH * stride_scale * stride_factor
        lift_height = STEP_HEIGHT * stride_scale
    else:
        stride = SPIN_STEP_LENGTH * stride_scale
        lift_height = SPIN_STEP_HEIGHT * stride_scale

    # 摆动相
    if leg_phase < SWING_RATIO:
        swing_progress = leg_phase / SWING_RATIO
        progress = smooth_step(swing_progress)

        forward_offset = (
            -stride / 2.0
            + stride * progress
        )

        vertical_offset = (
            lift_height * math.sin(math.pi * swing_progress)
        )

    # 支撑相
    else:
        stance_progress = (
            (leg_phase - SWING_RATIO)
            / (1.0 - SWING_RATIO)
        )

        progress = smooth_step(stance_progress)

        forward_offset = (
            stride / 2.0
            - stride * progress
        )

        vertical_offset = 0.0

    # 普通行走：完全保留原来的前后足端轨迹
    if spin == 0:
        local_target = [
            nominal_foot_positions[leg][axis]
            + forward_vector[axis] * forward_offset
            for axis in range(3)
        ]

    # 原地旋转：根据每只脚相对于机身中心的位置
    # 计算切向运动方向
    else:
        foot = nominal_foot_positions[leg]

        forward_coordinate = sum(
            foot[i] * forward_vector[i]
            for i in range(3)
        )

        lateral_coordinate = sum(
            foot[i] * lateral_vector[i]
            for i in range(3)
        )

        # Q：spin=1，左转
        # E：spin=-1，右转
        tangent = [
            spin * (
                -lateral_coordinate * forward_vector[i]
                + forward_coordinate * lateral_vector[i]
            )
            for i in range(3)
        ]

        tangent_length = math.sqrt(
            sum(value * value for value in tangent)
        )

        if tangent_length > 1e-6:
            tangent = [
                value / tangent_length
                for value in tangent
            ]
        else:
            tangent = forward_vector

        local_target = [
            foot[i] + tangent[i] * forward_offset
            for i in range(3)
        ]

    # 转换为世界坐标系
    current_base_position, current_base_orientation = (
        p.getBasePositionAndOrientation(robot_id)
    )

    world_target, _ = p.multiplyTransforms(
        current_base_position,
        current_base_orientation,
        local_target,
        [0, 0, 0, 1]
    )

    # 抬腿方向使用世界坐标系Z轴
    world_target = [
        world_target[0],
        world_target[1],
        world_target[2] + vertical_offset
    ]

    return world_target

# ==========================================
# 13. 使用带参考姿态的逆运动学控制单腿
# ==========================================

def control_leg_ik(leg, target_position):

    rest_poses = get_ik_rest_poses()

    lower_limits = []
    upper_limits = []
    joint_ranges = []

    for joint_index in movable_joint_indices:
        info = p.getJointInfo(robot_id, joint_index)

        lower = info[8]
        upper = info[9]

        if upper <= lower:
            lower = -math.pi
            upper = math.pi

        lower_limits.append(lower)
        upper_limits.append(upper)
        joint_ranges.append(upper - lower)

    solution = p.calculateInverseKinematics(
        bodyUniqueId=robot_id,
        endEffectorLinkIndex=toe_links[leg],
        targetPosition=target_position,
        lowerLimits=lower_limits,
        upperLimits=upper_limits,
        jointRanges=joint_ranges,
        restPoses=rest_poses,
        maxNumIterations=80,
        residualThreshold=1e-4
    )

    if len(solution) < len(movable_joint_indices):
        raise RuntimeError(
            f"{leg} 腿的逆运动学结果不完整"
        )

    target_angles = [
        solution[ik_joint_map[joint_index]]
        for joint_index in leg_groups[leg]
    ]

    set_leg_angles(
        leg,
        target_angles,
        force=100
    )

# ==========================================
# 14. 四足步态控制
# ==========================================

def control_gait(phase, stride_scale, turn, spin=0):

    for leg in ["FR", "FL", "RR", "RL"]:
        target_position = calculate_foot_target(
            leg,
            phase,
            stride_scale,
            turn,
            spin
        )

        control_leg_ik(
            leg,
            target_position
        )

# ==========================================
# 15. 键盘控制说明
# ==========================================

print("\n========== Laikago 足端轨迹控制 ==========")
print("W：前进")
print("S：后退")
print("A：左转")
print("D：右转")
print("Q：原地左转")
print("E：原地右转")
print("W + A：向前左转")
print("W + D：向前右转")
print("S + A：向后左转")
print("S + D：向后右转")
print("空格：停止并站立")
print("==========================================")
print("注意：先测试站立，再短按W、A、D、Q、E。")

# ==========================================
# 16. 主循环
# ==========================================

last_time = time.time()
last_print_time = time.time()

walk_mode = 0
turn_mode = 0
spin_mode = 0

control_accumulator = 0.0
motion_blend = 0.0

try:
    while True:

        keys = p.getKeyboardEvents()

        def is_down(key):
            return (
                key in keys
                and bool(keys[key] & p.KEY_IS_DOWN)
            )

        w_down = is_down(ord("w"))
        s_down = is_down(ord("s"))
        a_down = is_down(ord("a"))
        d_down = is_down(ord("d"))

        q_down = is_down(ord("q"))
        e_down = is_down(ord("e"))

        space_key = getattr(p, "B3G_SPACE", 32)
        space_down = is_down(space_key)

        # 空格键优先停止
        if space_down:
            walk_mode = 0
            turn_mode = 0
            spin_mode = 0

        else:
            # 前进和后退
            if w_down:
                walk_mode = 1
            elif s_down:
                walk_mode = -1
            else:
                walk_mode = 0

            # A/D 行进转向
            if a_down:
                turn_mode = 1
            elif d_down:
                turn_mode = -1
            else:
                turn_mode = 0

            # Q/E 原地旋转
            # 有 W/S 指令时，优先执行前进或后退
            if q_down and not w_down and not s_down:
                spin_mode = 1
            elif e_down and not w_down and not s_down:
                spin_mode = -1
            else:
                spin_mode = 0

        # 计算时间间隔
        now = time.time()

        dt = min(
            now - last_time,
            0.02
        )

        last_time = now
        control_accumulator += dt

        # 固定控制频率
        if control_accumulator >= CONTROL_DT:

            control_accumulator -= CONTROL_DT

            if (
                walk_mode != 0
                or turn_mode != 0
                or spin_mode != 0
            ):

                motion_blend = min(
                    1.0,
                    motion_blend + CONTROL_DT / RAMP_TIME
                )

                # 原地旋转使用独立周期
                if spin_mode != 0:
                    gait_phase += (
                        CONTROL_DT / SPIN_PERIOD
                    )

                    active_turn = 0

                # 普通前进、后退和行进转向
                else:
                    direction = walk_mode

                    if direction == 0:
                        direction = 1

                    gait_phase += (
                        direction * CONTROL_DT / GAIT_PERIOD
                    )

                    active_turn = turn_mode

                gait_phase %= 1.0

                control_gait(
                    gait_phase,
                    motion_blend,
                    active_turn,
                    spin_mode
                )

            else:
                motion_blend = 0.0
                stand()

        # 物理仿真
        p.stepSimulation()

        # 每两秒打印机身状态
        if now - last_print_time >= 2.0:

            position, orientation_now = (
                p.getBasePositionAndOrientation(robot_id)
            )

            roll, pitch, yaw = p.getEulerFromQuaternion(
                orientation_now
            )

            print(
                "机身位置:",
                [round(value, 3) for value in position],

                "姿态角(度):",
                [
                    round(math.degrees(roll), 2),
                    round(math.degrees(pitch), 2),
                    round(math.degrees(yaw), 2)
                ],

                "运动模式:",
                walk_mode,

                "转向模式:",
                turn_mode,

                "原地旋转:",
                spin_mode
            )

            last_print_time = now

        time.sleep(1.0 / 240.0)

except KeyboardInterrupt:
    print("\n正在关闭仿真...")

finally:
    p.disconnect()