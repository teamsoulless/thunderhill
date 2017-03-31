from ctypes import *
lib = cdll.LoadLibrary('./MainNode/libMainNode.so')

class MainNode(object):
    def __init__(self, imagefunc):
        self.obj = lib.MainNode_new()
        self.imagefunc = imagefunc

        # set img callback
        self.FUNC1 = CFUNCTYPE(None, c_int, POINTER(c_ubyte), c_float, c_float, c_float)
        self.func1 = self.FUNC1(self.imagefunc)
        lib.MainNode_setImageCallback(self.obj, self.func1)

    def steerCommand(self, angle):
    	lib.MainNode_steerCommand(self.obj, angle)

    def brakeCommand(self, value):
        lib.MainNode_brakeCommand(self.obj, value)

    def throttleCommand(self, value):
        lib.MainNode_throttleCommand(self.obj, value)

    def connectPolySync(self):
        lib.MainNode_connectPolySync(self.obj)
