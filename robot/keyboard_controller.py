
import pybullet as p


class KeyboardController:
    """统一的 PyBullet 键盘控制接口，持续记录按键状态"""

    def __init__(self):
        self.pressed_keys = set()

    def get_command(self):

        # 获取本帧键盘事件
        keys = p.getKeyboardEvents()

        # 更新持续按键状态
        for key, event in keys.items():

            # 按键处于按下状态
            if event & p.KEY_IS_DOWN:
                self.pressed_keys.add(key)

            # 按键已经释放
            elif event & p.KEY_WAS_RELEASED:
                self.pressed_keys.discard(key)

        # ESC：退出
        if 27 in self.pressed_keys or 65307 in self.pressed_keys:
            return {
                "walk": 0,
                "turn": 0,
                "spin": 0,
                "stop": True,
                "exit": True
            }

        # 空格：停止
        if 32 in self.pressed_keys:
            return {
                "walk": 0,
                "turn": 0,
                "spin": 0,
                "stop": True,
                "exit": False
            }

        # 前进、后退
        w = ord("w") in self.pressed_keys
        s = ord("s") in self.pressed_keys

        walk = 1 if w else (-1 if s else 0)

        # 左转、右转
        a = ord("a") in self.pressed_keys
        d = ord("d") in self.pressed_keys

        turn = 1 if a else (-1 if d else 0)

        # 原地旋转
        q = ord("q") in self.pressed_keys
        e = ord("e") in self.pressed_keys

        spin = 0

        if walk == 0:
            spin = 1 if q else (-1 if e else 0)

        return {
            "walk": walk,
            "turn": turn,
            "spin": spin,
            "stop": False,
            "exit": False
        }
