
from robot.keyboard_controller import KeyboardController
import time
import pybullet as p

from robot.robot_controller import RobotController


def main():
    robot = RobotController()
    keyboard = KeyboardController()

    try:
        # 启动机器人仿真和控制器
        robot.start()

        print("\n========== AI Virtual Robot Assistant ==========")
        print("W：前进")
        print("S：后退")
        print("A：左转")
        print("D：右转")
        print("Q：原地左旋转")
        print("E：原地右旋转")
        print("空格：停止")
        print("ESC：退出")
        print("================================================")

        while p.isConnected():

            # 使用公共键盘控制接口读取按键
            command = keyboard.get_command()

            # ESC：退出
            if command["exit"]:
                break

            # 空格：停止
            if command["stop"]:
                robot.stop()

            else:
                # 将运动指令交给统一控制器
                robot.set_motion(
                    walk=command["walk"],
                    turn=command["turn"],
                    spin=command["spin"]
                )

            # 更新机器人运动和物理仿真
            robot.update()

            time.sleep(1.0 / 240.0)

    except KeyboardInterrupt:
        print("\n用户中断仿真。")

    finally:
        robot.close()
        print("机器人仿真已关闭。")


if __name__ == "__main__":
    main()
