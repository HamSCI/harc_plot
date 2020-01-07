from .omni import Omni

class GeospaceEnv(object):
    def __init__(self):
        self.omni   = Omni()
        self.symh   = self.omni.symh
