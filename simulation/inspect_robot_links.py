
import pybullet as p
import pybullet_data
import time
import math

# ==========================================
# 1. 创建 PyBullet 仿真环境
# ==========================================

physics_client = p.connect(p.GUI)

p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setAdditionalSearchPath(r"C:\pybullet_data")

p.setGravity(0, 0, -9.8)
p.setTimeStep(1.0 / 240.0)
p.setRealTimeSimulation(0)

# 加载地面
plane = p.loadURDF("plane.urdf")

# ==========================================
# 2. 加载与 main.py 相同的 Laikago 模型
# ==========================================

robot = p.loadURDF(
    r"C:\pybullet_data\laikago\laikago_toes.urdf",
    basePosition=[0, 0, 0.50],
    baseOrientation=p.getQuaternionFromEuler(
        [1.5707963, 0, 0]
    ),
    useFixedBase=False
)

print("\n========== Laikago 模型结构 ==========")

# ==========================================
# 3. 查看机身信息
# ==========================================

base_name = p.getBodyInfo(robot)[0].decode("utf-8")

print("\n机身 Base Link:")
print(base_name)

# ==========================================
# 4. 查看所有关节和连杆
# ==========================================

print("\n========== 关节与连杆列表 ==========")

for i in range(p.getNumJoints(robot)):

    joint_info = p.getJointInfo(robot, i)

    joint_name = joint_info[1].decode("utf-8")
    link_name = joint_info[12].decode("utf-8")

    print(
        f"关节编号: {i:2d} | "
        f"关节名称: {joint_name:20s} | "
        f"连杆名称: {link_name}"
    )

print("\n========== 检查完成 ==========")
print("请查看上面的机身、关节和连杆名称。")

# 保持仿真窗口打开
while p.isConnected():
    p.stepSimulation()
    time.sleep(1.0 / 240.0)
