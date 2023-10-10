from .omni import Omni

class GeospaceEnv(object):
#    def __init__(self):
#        self.omni   = Omni()
    def __init__(self, years=[2016,2017]):     # Add year parameter. Kukkai, 20231002
        self.omni   = Omni(years=years)        # Pass year parameter. Kukkai, 20231002
        self.symh   = self.omni.symh
