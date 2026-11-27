
import time
import pybullet as p

from robot.robot_controller import RobotController


robot = RobotController()


def key_down(keys, key):
    return (
        key in keys
        and keys[key] & p.KEY_IS_DOWN
    )


try:

    robot.start()

    print("\n========== 键盘控制 ==========")
    print("W：前进")
    print("S：后退")
    print("A：左转")
    print("D：右转")
    print("Q：原地左转")
    print("E：原地右转")
    print("空格：停止")
    print("ESC：退出")

    while p.isConnected():

        keys = p.getKeyboardEvents()

        # ESC：退出
        if key_down(keys, 27):
            break

        # 空格：优先停止
        if key_down(keys, 32):

            robot.stop()

        else:

            w = key_down(keys, ord("w"))
            s = key_down(keys, ord("s"))

            a = key_down(keys, ord("a"))
            d = key_down(keys, ord("d"))

            q = key_down(keys, ord("q"))
            e = key_down(keys, ord("e"))

            # 前进和后退
            walk = 0

            if w:
                walk = 1
            elif s:
                walk = -1

            # 左转和右转
            turn = 0

            if a:
                turn = 1
            elif d:
                turn = -1

            # 原地旋转
            spin = 0

            if walk == 0:

                if q:
                    spin = 1
                elif e:
                    spin = -1

            robot.set_motion(
                walk=walk,
                turn=turn,
                spin=spin
            )

        robot.update()

        time.sleep(1.0 / 240.0)

finally:

    robot.close()
