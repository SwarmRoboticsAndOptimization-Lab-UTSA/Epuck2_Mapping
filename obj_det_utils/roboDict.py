import threading

class RobotDict:
    def __init__(self):
        self.robot_dic = {}
        self.lock = threading.Lock()

    def get(self, key):
        with self.lock:
            return self.robot_dic.get(key, None)

    def set(self, key, value):
        with self.lock:
            self.robot_dic[key] = value

    def get_all(self):
        with self.lock:
            return self.robot_dic.copy()
