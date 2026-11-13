from gpustack.worker.backends.base import InferenceServer


class CustomServer(InferenceServer):
    """第 19 章换成完整版本。"""

    def start(self):
        raise NotImplementedError("Custom backends are covered in chapter 19.")
