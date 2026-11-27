
import time
import math
import pybullet as p


class RobotController:

    def __init__(self):

        self.robot_id = None
        self.physics_client = None

        self.leg_groups = {
            "FR": [],
            "FL": [],
            "RR": [],
            "RL": []
        }

        self.toe_links = {
            "FR": 3,
            "FL": 7,
            "RR": 11,
            "RL": 15
        }

        self.default_angles = [0.0, 0.7, -1.4]

        self.movable_joint_indices = []
        self.ik_joint_map = {}

        self.nominal_foot_positions = {}

        self.forward_vector = []
        self.lateral_vector = []

        # 步态参数：保留之前测试满意的版本
        self.GAIT_PERIOD = 1.6
        self.SWING_RATIO = 0.25
        self.STEP_LENGTH = 0.08
        self.STEP_HEIGHT = 0.035

        self.TURN_GAIN = 0.35

        self.SPIN_PERIOD = 1.8
        self.SPIN_STEP_LENGTH = 0.055
        self.SPIN_STEP_HEIGHT = 0.025

        self.leg_phase_offset = {
            "FR": 0.00,
            "RL": 0.25,
            "FL": 0.50,
            "RR": 0.75
        }

        self.RAMP_TIME = 1.0
        self.CONTROL_DT = 1.0 / 60.0

        self.gait_phase = 0.0
        self.motion_blend = 0.0

        self.walk_mode = 0
        self.turn_mode = 0
        self.spin_mode = 0

        self.last_time = time.time()
        self.control_accumulator = 0.0

        self.started = False

        print("RobotController 创建成功")

    # ======================================
    # 1. 启动仿真
    # ======================================

    def start(self):

        if self.started:
            print("机器人已经启动")
            return

        self.physics_client = p.connect(p.GUI)

        if self.physics_client < 0:
            self.physics_client = None
            raise RuntimeError("PyBullet 启动失败")

        try:

            # 使用原项目的 PyBullet 数据目录
            p.setAdditionalSearchPath(
                r"C:\pybullet_data"
            )

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

            orientation = p.getQuaternionFromEuler(
                [1.5707963, 0, 0]
            )

            self.robot_id = p.loadURDF(
                r"C:\pybullet_data\laikago\laikago_toes.urdf",
                basePosition=[0, 0, 0.50],
                baseOrientation=orientation,
                useFixedBase=False
            )

            if self.robot_id < 0:
                raise RuntimeError("Laikago 加载失败")

            p.resetDebugVisualizerCamera(
                cameraDistance=2.5,
                cameraYaw=45,
                cameraPitch=-20,
                cameraTargetPosition=[0, 0, 0.3]
            )

            self._initialize_joints()

            # 先让机器人落地稳定
            for _ in range(240):
                p.stepSimulation()
                time.sleep(1.0 / 240.0)

            self._initialize_feet()
            self._calculate_directions()

            self.started = True
            self.last_time = time.time()
            self.control_accumulator = 0.0

            print("Laikago 控制器初始化成功")

        except Exception:

            if self.physics_client is not None:
                if p.isConnected(self.physics_client):
                    p.disconnect(self.physics_client)

            self.physics_client = None
            self.robot_id = None
            self.started = False

            raise

    # ======================================
    # 2. 识别关节
    # ======================================

    def _initialize_joints(self):

        for joint_index in range(
            p.getNumJoints(self.robot_id)
        ):

            info = p.getJointInfo(
                self.robot_id,
                joint_index
            )

            joint_name = info[1].decode("utf-8")

            if info[2] == p.JOINT_REVOLUTE:

                for leg in self.leg_groups:

                    if joint_name.startswith(leg):
                        self.leg_groups[leg].append(
                            joint_index
                        )

        print("\n========== 四条腿识别结果 ==========")

        for leg, joints in self.leg_groups.items():

            print(leg, joints)

            if len(joints) != 3:
                raise RuntimeError(
                    "{} 腿应有3个旋转关节，实际识别到 {}".format(
                        leg, len(joints)
                    )
                )

        all_joints = [
            joint
            for leg in ["FR", "FL", "RR", "RL"]
            for joint in self.leg_groups[leg]
        ]

        if len(set(all_joints)) != 12:
            raise RuntimeError("四条腿关节重复或数量不正确")

        for joint_index in range(
            p.getNumJoints(self.robot_id)
        ):

            info = p.getJointInfo(
                self.robot_id,
                joint_index
            )

            if info[2] != p.JOINT_FIXED:
                self.movable_joint_indices.append(
                    joint_index
                )

        if len(self.movable_joint_indices) != 12:
            raise RuntimeError(
                "可运动关节数量不是12，实际数量：{}".format(
                    len(self.movable_joint_indices)
                )
            )

        self.ik_joint_map = {
            joint: index
            for index, joint in enumerate(
                self.movable_joint_indices
            )
        }

        # 设置初始关节角度

        for leg in ["FR", "FL", "RR", "RL"]:

            for joint_index, angle in zip(
                self.leg_groups[leg],
                self.default_angles
            ):

                p.resetJointState(
                    self.robot_id,
                    joint_index,
                    angle
                )

        self.stand()

        # 设置脚掌摩擦力

        for link_index in self.toe_links.values():

            p.changeDynamics(
                self.robot_id,
                link_index,
                lateralFriction=1.5,
                spinningFriction=0.02
            )

        print("关节初始化完成")

    # ======================================
    # 3. 设置单条腿的关节角度
    # ======================================

    def _set_leg_angles(
        self,
        leg,
        angles,
        force=100
    ):

        p.setJointMotorControlArray(
            bodyUniqueId=self.robot_id,
            jointIndices=self.leg_groups[leg],
            controlMode=p.POSITION_CONTROL,
            targetPositions=angles,
            forces=[force] * 3,
            positionGains=[0.2] * 3,
            velocityGains=[1.0] * 3
        )

    # ======================================
    # 4. 站立
    # ======================================

    def stand(self):

        if self.robot_id is None:
            return

        for leg in ["FR", "FL", "RR", "RL"]:

            self._set_leg_angles(
                leg,
                self.default_angles
            )

    # ======================================
    # 5. 获取初始脚掌位置
    # ======================================

    def _initialize_feet(self):

        base_position, base_orientation = (
            p.getBasePositionAndOrientation(
                self.robot_id
            )
        )

        inverse_base_position, inverse_base_orientation = (
            p.invertTransform(
                base_position,
                base_orientation
            )
        )

        for leg, link_index in self.toe_links.items():

            link_state = p.getLinkState(
                self.robot_id,
                link_index,
                computeForwardKinematics=True
            )

            foot_world_position = link_state[4]

            foot_local_position, _ = p.multiplyTransforms(
                inverse_base_position,
                inverse_base_orientation,
                foot_world_position,
                [0, 0, 0, 1]
            )

            self.nominal_foot_positions[leg] = list(
                foot_local_position
            )

            print(
                leg,
                "初始脚掌位置:",
                [
                    round(value, 4)
                    for value in foot_local_position
                ]
            )

    # ======================================
    # 6. 计算机身前进和左右方向
    # ======================================

    def _calculate_directions(self):

        front_center = [
            (
                self.nominal_foot_positions["FR"][axis]
                + self.nominal_foot_positions["FL"][axis]
            ) / 2.0
            for axis in range(3)
        ]

        rear_center = [
            (
                self.nominal_foot_positions["RR"][axis]
                + self.nominal_foot_positions["RL"][axis]
            ) / 2.0
            for axis in range(3)
        ]

        self.forward_vector = [
            front_center[axis] - rear_center[axis]
            for axis in range(3)
        ]

        length = math.sqrt(
            sum(value * value for value in self.forward_vector)
        )

        if length < 0.001:
            raise RuntimeError("无法确定机身前进方向")

        self.forward_vector = [
            value / length
            for value in self.forward_vector
        ]

        left_center = [
            (
                self.nominal_foot_positions["FL"][axis]
                + self.nominal_foot_positions["RL"][axis]
            ) / 2.0
            for axis in range(3)
        ]

        right_center = [
            (
                self.nominal_foot_positions["FR"][axis]
                + self.nominal_foot_positions["RR"][axis]
            ) / 2.0
            for axis in range(3)
        ]

        self.lateral_vector = [
            left_center[i] - right_center[i]
            for i in range(3)
        ]

        projection = sum(
            self.lateral_vector[i] * self.forward_vector[i]
            for i in range(3)
        )

        self.lateral_vector = [
            self.lateral_vector[i]
            - projection * self.forward_vector[i]
            for i in range(3)
        ]

        length = math.sqrt(
            sum(value * value for value in self.lateral_vector)
        )

        if length < 0.001:
            raise RuntimeError("无法确定机身左右方向")

        self.lateral_vector = [
            value / length
            for value in self.lateral_vector
        ]

        print("机身前进方向:", self.forward_vector)
        print("机身左右方向:", self.lateral_vector)

    # ======================================
    # 7. 平滑插值
    # ======================================

    @staticmethod
    def _smooth_step(value):

        value = max(0.0, min(1.0, value))

        return (
            3.0 * value * value
            - 2.0 * value * value * value
        )

    # ======================================
    # 8. 获取IK参考角度
    # ======================================

    def _get_ik_rest_poses(self):

        rest_poses = []

        for joint_index in self.movable_joint_indices:

            joint_state = p.getJointState(
                self.robot_id,
                joint_index
            )

            rest_poses.append(joint_state[0])

        return rest_poses

    # ======================================
    # 9. 计算脚掌目标位置
    # ======================================

    def _calculate_foot_target(
        self,
        leg,
        phase,
        stride_scale,
        turn,
        spin=0
    ):

        if turn > 0:

            stride_factor = (
                1.0 + self.TURN_GAIN
                if leg in ["FR", "RR"]
                else 1.0 - self.TURN_GAIN
            )

        elif turn < 0:

            stride_factor = (
                1.0 + self.TURN_GAIN
                if leg in ["FL", "RL"]
                else 1.0 - self.TURN_GAIN
            )

        else:
            stride_factor = 1.0

        leg_phase = (
            phase - self.leg_phase_offset[leg]
        ) % 1.0

        if spin == 0:

            stride = (
                self.STEP_LENGTH
                * stride_scale
                * stride_factor
            )

            lift_height = (
                self.STEP_HEIGHT
                * stride_scale
            )

        else:

            stride = (
                self.SPIN_STEP_LENGTH
                * stride_scale
            )

            lift_height = (
                self.SPIN_STEP_HEIGHT
                * stride_scale
            )

        # 摆动相

        if leg_phase < self.SWING_RATIO:

            swing_progress = (
                leg_phase / self.SWING_RATIO
            )

            progress = self._smooth_step(
                swing_progress
            )

            forward_offset = (
                -stride / 2.0
                + stride * progress
            )

            vertical_offset = (
                lift_height
                * math.sin(math.pi * swing_progress)
            )

        # 支撑相

        else:

            stance_progress = (
                (leg_phase - self.SWING_RATIO)
                / (1.0 - self.SWING_RATIO)
            )

            progress = self._smooth_step(
                stance_progress
            )

            forward_offset = (
                stride / 2.0
                - stride * progress
            )

            vertical_offset = 0.0

        # 普通行走

        if spin == 0:

            local_target = [
                self.nominal_foot_positions[leg][axis]
                + self.forward_vector[axis] * forward_offset
                for axis in range(3)
            ]

        # 原地旋转

        else:

            foot = self.nominal_foot_positions[leg]

            forward_coordinate = sum(
                foot[i] * self.forward_vector[i]
                for i in range(3)
            )

            lateral_coordinate = sum(
                foot[i] * self.lateral_vector[i]
                for i in range(3)
            )

            tangent = [
                spin * (
                    -lateral_coordinate * self.forward_vector[i]
                    + forward_coordinate * self.lateral_vector[i]
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
                tangent = self.forward_vector

            local_target = [
                foot[i] + tangent[i] * forward_offset
                for i in range(3)
            ]

        current_base_position, current_base_orientation = (
            p.getBasePositionAndOrientation(
                self.robot_id
            )
        )

        world_target, _ = p.multiplyTransforms(
            current_base_position,
            current_base_orientation,
            local_target,
            [0, 0, 0, 1]
        )

        world_target = [
            world_target[0],
            world_target[1],
            world_target[2] + vertical_offset
        ]

        return world_target

    # ======================================
    # 10. IK控制单腿
    # ======================================

    def _control_leg_ik(
        self,
        leg,
        target_position
    ):

        rest_poses = self._get_ik_rest_poses()

        lower_limits = []
        upper_limits = []
        joint_ranges = []

        for joint_index in self.movable_joint_indices:

            info = p.getJointInfo(
                self.robot_id,
                joint_index
            )

            lower = info[8]
            upper = info[9]

            if upper <= lower:
                lower = -math.pi
                upper = math.pi

            lower_limits.append(lower)
            upper_limits.append(upper)
            joint_ranges.append(upper - lower)

        solution = p.calculateInverseKinematics(
            bodyUniqueId=self.robot_id,
            endEffectorLinkIndex=self.toe_links[leg],
            targetPosition=target_position,
            lowerLimits=lower_limits,
            upperLimits=upper_limits,
            jointRanges=joint_ranges,
            restPoses=rest_poses,
            maxNumIterations=80,
            residualThreshold=1e-4
        )

        if len(solution) < len(self.movable_joint_indices):
            raise RuntimeError(
                "{} 腿的逆运动学结果不完整".format(leg)
            )

        target_angles = [
            solution[self.ik_joint_map[joint_index]]
            for joint_index in self.leg_groups[leg]
        ]

        self._set_leg_angles(
            leg,
            target_angles,
            force=100
        )

    # ======================================
    # 11. 四足步态控制
    # ======================================

    def _control_gait(
        self,
        phase,
        stride_scale,
        turn,
        spin=0
    ):

        for leg in ["FR", "FL", "RR", "RL"]:

            target_position = self._calculate_foot_target(
                leg,
                phase,
                stride_scale,
                turn,
                spin
            )

            self._control_leg_ik(
                leg,
                target_position
            )

    # ======================================
    # 12. 对外运动接口
    # ======================================

    def set_motion(
        self,
        walk=0,
        turn=0,
        spin=0
    ):

        if not self.started:
            raise RuntimeError("请先调用 start()")

        self.walk_mode = max(-1, min(1, walk))
        self.turn_mode = max(-1, min(1, turn))
        self.spin_mode = max(-1, min(1, spin))

        # W/S 优先于 Q/E
        if self.walk_mode != 0:
            self.spin_mode = 0

    def forward(self):
        self.set_motion(walk=1)

    def backward(self):
        self.set_motion(walk=-1)

    def turn_left(self):
        self.set_motion(walk=1, turn=1)

    def turn_right(self):
        self.set_motion(walk=1, turn=-1)

    def spin_left(self):
        self.set_motion(spin=1)

    def spin_right(self):
        self.set_motion(spin=-1)

    def stop(self):
        self.set_motion()

    # ======================================
    # 13. 更新运动和物理仿真
    # ======================================

    def update(self):

        if not self.started:
            return

        now = time.time()

        dt = min(
            now - self.last_time,
            0.02
        )

        self.last_time = now
        self.control_accumulator += dt

        if self.control_accumulator >= self.CONTROL_DT:

            self.control_accumulator -= self.CONTROL_DT

            if (
                self.walk_mode != 0
                or self.turn_mode != 0
                or self.spin_mode != 0
            ):

                self.motion_blend = min(
                    1.0,
                    self.motion_blend
                    + self.CONTROL_DT / self.RAMP_TIME
                )

                if self.spin_mode != 0:

                    self.gait_phase += (
                        self.CONTROL_DT / self.SPIN_PERIOD
                    )

                    active_turn = 0

                else:

                    direction = self.walk_mode

                    if direction == 0:
                        direction = 1

                    self.gait_phase += (
                        direction
                        * self.CONTROL_DT
                        / self.GAIT_PERIOD
                    )

                    active_turn = self.turn_mode

                self.gait_phase %= 1.0

                self._control_gait(
                    self.gait_phase,
                    self.motion_blend,
                    active_turn,
                    self.spin_mode
                )

            else:

                self.motion_blend = 0.0
                self.stand()

        p.stepSimulation()

    # ======================================
    # 14. 读取机器人状态
    # ======================================

    def get_status(self):

        if not self.started:
            return None

        position, orientation = (
            p.getBasePositionAndOrientation(
                self.robot_id
            )
        )

        roll, pitch, yaw = p.getEulerFromQuaternion(
            orientation
        )

        return {
            "position": position,
            "roll": math.degrees(roll),
            "pitch": math.degrees(pitch),
            "yaw": math.degrees(yaw)
        }

    # ======================================
    # 15. 关闭仿真
    # ======================================

    def close(self):

        if self.physics_client is not None:

            if p.isConnected(self.physics_client):
                p.disconnect(self.physics_client)

            self.physics_client = None
            self.robot_id = None
            self.started = False

            print("RobotController 已关闭")
