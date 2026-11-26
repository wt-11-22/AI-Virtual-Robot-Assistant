# AI-Virtual-Robot-Assistant

**English | [简体中文](#简体中文)**

An open-source virtual quadruped robot project built with Python and PyBullet.

The goal of this project is to develop an intelligent virtual robot dog with locomotion control, visual perception, voice interaction, and AI-powered conversation capabilities.

The project is currently under active development. The current implementation includes quadruped robot simulation, keyboard-based movement control, virtual camera sensing, depth image capture, and YOLO-based object detection.

## Current Features

### 1. Quadruped Robot Simulation

- Physics simulation using PyBullet
- Laikago quadruped robot model
- Basic standing posture control
- Experimental quadruped gait control
- Keyboard-based movement control
- Forward, backward, and turning commands
- Stop and return to standing posture using the Space key

### 2. Camera and Visual Perception

- Virtual camera based on PyBullet
- RGB image capture
- Depth image capture
- YOLOv8n object detection using ONNX Runtime
- Object detection in the simulated environment

### 3. 3D Environment Assets

- Plant model for the simulation environment
- OBJ and GLB model assets
- Material and texture resources

**Note:** The current locomotion controller is experimental. Walking stability and movement performance still need further improvement.

## Requirements

- Operating System: Windows
- Python: 3.8
- Physics Simulation: PyBullet
- Computer Vision: OpenCV
- Numerical Computing: NumPy
- Model Inference: ONNX Runtime

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/wt-11-22/AI-Virtual-Robot-Assistant.git
cd AI-Virtual-Robot-Assistant
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Create a Python Virtual Environment

```bash
python -m venv .venv
```

Activate the environment in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

## Usage

The project currently provides two runnable simulation programs. Both programs support robot movement control.

### 1. Basic Robot Simulation

Run:

```bash
python main.py
```

Features:

- Laikago quadruped robot simulation
- Keyboard-based locomotion control
- Forward, backward, and turning movements
- Stop and return to standing posture

### 2. Robot Simulation with Visual Perception

Run:

```bash
python simulation/camera.py
```

Features:

- Laikago quadruped robot simulation
- Keyboard-based locomotion control
- RGB camera image capture
- Depth image capture
- YOLOv8n object detection using ONNX Runtime

The camera version provides the same basic robot movement control, with additional visual perception capabilities.

### Keyboard Controls

| Key | Action |
|---|---|
| W | Move Forward |
| S | Move Backward |
| A | Turn Left |
| D | Turn Right |
| Space | Stop and Return to Standing Posture |

## Robot Model

The current robot simulation uses the Laikago URDF model provided by PyBullet.

The current development environment uses the following model path:

```text
C:\pybullet_data\laikago\laikago_toes.urdf
```

Make sure the required PyBullet data resources are available on your computer.

**Note:** The current implementation uses an absolute Windows path. A more configurable robot model path will be introduced in a future update.

## Project Structure

```text
AI-Virtual-Robot-Assistant/
│
├── assets/
│   └── plant/
│       ├── material.mtl
│       ├── material_0.png
│       ├── material_1.png
│       ├── plant.glb
│       ├── plant.obj
│       └── plant_pybullet.obj
│
├── simulation/
│   ├── camera.py
│   └── inspect_robot_links.py
│
├── vision/
│   ├── models/
│   │   └── yolov8n.onnx
│   └── object_detection.py
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

The `.venv` directory is a local Python virtual environment and is excluded from Git.

## Development Roadmap

### Phase 1: Basic Simulation

- [x] PyBullet simulation environment
- [x] Laikago robot model loading
- [x] Basic standing posture control
- [x] Keyboard movement control
- [ ] Improve locomotion stability
- [ ] Add more movement modes
- [ ] Improve robot state monitoring

### Phase 2: Visual Perception

- [x] Virtual camera integration
- [x] RGB image capture
- [x] Depth image capture
- [x] YOLOv8n ONNX object detection
- [ ] Improve object recognition
- [ ] Integrate visual perception with robot behavior

### Phase 3: Voice Interaction

- [ ] Speech recognition
- [ ] Voice command processing
- [ ] Voice-based robot control
- [ ] Voice feedback

### Phase 4: AI-Powered Interaction

- [ ] Integrate a large language model (LLM)
- [ ] Natural language conversation
- [ ] Natural language robot control
- [ ] Connect vision, voice, and locomotion systems

## Contributing

Contributions are welcome!

You can contribute by:

- Opening an Issue to report bugs or suggest improvements
- Submitting a Pull Request with code improvements
- Improving quadruped locomotion algorithms
- Developing new vision, voice, and AI features

Let's work together to make this virtual robot dog smarter and more capable!

## License

The project license will be determined before the official public release.

## Project Information

- **Project:** AI-Virtual-Robot-Assistant
- **Version:** v0.1.0
- **Status:** Early Development
- **Language:** Python
- **Simulation:** PyBullet

---

# 简体中文

一个基于 Python 和 PyBullet 开发的开源虚拟四足机器狗项目。

本项目旨在开发一只具有运动控制、视觉感知、语音交互和 AI 对话能力的智能虚拟机器狗。

项目目前仍处于积极开发阶段。现阶段已经实现四足机器人仿真、键盘运动控制、虚拟摄像头、深度图像获取以及基于 YOLO 的目标检测。

## 当前功能

### 1. 四足机器人仿真

- 使用 PyBullet 进行物理仿真
- 加载 Laikago 四足机器人模型
- 基础站立姿态控制
- 实验性的四足步态控制
- 键盘运动控制
- 前进、后退和转向控制
- 使用空格键停止运动并恢复站立姿态

### 2. 摄像头与视觉感知

- 基于 PyBullet 的虚拟摄像头
- RGB 彩色图像获取
- 深度图像获取
- 使用 ONNX Runtime 运行 YOLOv8n 目标检测模型
- 在仿真环境中进行物体识别

### 3. 三维环境资源

- 用于仿真环境的植物模型
- OBJ 和 GLB 三维模型资源
- 模型材质与纹理文件

**注意：** 当前机器人运动控制器仍处于实验阶段，行走稳定性和运动表现还有待进一步优化。

## 环境要求

- 操作系统：Windows
- Python：3.8
- 物理仿真：PyBullet
- 计算机视觉：OpenCV
- 数值计算：NumPy
- 模型推理：ONNX Runtime

## 安装步骤

### 1. 克隆项目仓库

```bash
git clone https://github.com/YOUR_USERNAME/AI-Virtual-Robot-Assistant.git
cd AI-Virtual-Robot-Assistant
```

将 `YOUR_USERNAME` 替换为你自己的 GitHub 用户名。

### 2. 创建 Python 虚拟环境

```bash
python -m venv .venv
```

在 Windows PowerShell 中激活虚拟环境：

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. 安装项目依赖

```bash
python -m pip install -r requirements.txt
```

## 运行项目

目前项目提供两个可以独立运行的仿真程序，两个程序都支持机器人运动控制。

### 1. 基础机器人仿真

运行：

```bash
python main.py
```

主要功能：

- Laikago 四足机器人仿真
- 键盘运动控制
- 前进、后退和转向
- 停止运动并恢复站立姿态

### 2. 带视觉感知功能的机器人仿真

运行：

```bash
python simulation/camera.py
```

主要功能：

- Laikago 四足机器人仿真
- 键盘运动控制
- RGB 摄像头图像获取
- 深度图像获取
- 使用 YOLOv8n 进行物体检测

该版本同样支持机器人行走控制，并额外集成了视觉感知功能。

### 键盘控制说明

| 按键 | 功能 |
|---|---|
| W | 向前移动 |
| S | 向后移动 |
| A | 向左转向 |
| D | 向右转向 |
| Space | 停止运动并恢复站立姿态 |

## 机器人模型

当前项目使用 PyBullet 提供的 Laikago URDF 机器人模型。

当前开发环境中的模型路径为：

```text
C:\pybullet_data\laikago\laikago_toes.urdf
```

请确保电脑上已经具备相应的 PyBullet 数据资源，并且该文件存在。

**注意：** 当前版本使用 Windows 绝对路径，后续计划改为更加灵活的模型路径配置方式。

## 项目目录结构

```text
AI-Virtual-Robot-Assistant/
│
├── assets/
│   └── plant/
│       ├── material.mtl
│       ├── material_0.png
│       ├── material_1.png
│       ├── plant.glb
│       ├── plant.obj
│       └── plant_pybullet.obj
│
├── simulation/
│   ├── camera.py
│   └── inspect_robot_links.py
│
├── vision/
│   ├── models/
│   │   └── yolov8n.onnx
│   └── object_detection.py
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

`.venv` 是本地 Python 虚拟环境目录，不会上传到 GitHub。

## 后续开发计划

### 第一阶段：基础仿真

- [x] PyBullet 仿真环境
- [x] Laikago 机器人模型加载
- [x] 基础站立姿态控制
- [x] 键盘运动控制
- [ ] 提高机器人行走稳定性
- [ ] 增加更多运动模式
- [ ] 完善机器人状态监测

### 第二阶段：视觉感知

- [x] 集成虚拟摄像头
- [x] RGB 图像获取
- [x] 深度图像获取
- [x] YOLOv8n ONNX 目标检测
- [ ] 优化物体识别效果
- [ ] 将视觉感知与机器人行为控制结合

### 第三阶段：语音交互

- [ ] 集成语音识别
- [ ] 实现语音指令处理
- [ ] 通过语音控制机器人
- [ ] 实现语音反馈

### 第四阶段：AI 智能交互

- [ ] 集成大语言模型（LLM）
- [ ] 实现自然语言对话
- [ ] 实现自然语言机器人控制
- [ ] 打通视觉、语音和运动控制系统

## 参与贡献

欢迎开发者参与本项目！

你可以通过以下方式帮助项目发展：

- 提交 Issue，反馈问题或提出改进建议
- 提交 Pull Request，贡献代码
- 改进四足机器人运动控制算法
- 开发新的视觉、语音和 AI 功能

让我们一起让这只虚拟机器狗变得更加智能、更加有趣！

## 开源许可证

项目将在正式公开发布前确定开源许可证。

## 项目信息

- **项目名称：** AI-Virtual-Robot-Assistant
- **版本：** v0.1.0
- **项目状态：** 早期开发阶段
- **开发语言：** Python
- **仿真平台：** PyBullet
