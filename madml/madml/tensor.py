#import backend
import numpy as np

class tensor(np.ndarray):

    def __new__(cls, input_array, info=None):
        obj = np.asarray(input_array).view(cls)
        obj.info = info
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.info = getattr(obj, 'info', None)

    def __array_wrap__(self, out_arr, context=None):
        return super(tensor, self).__array_wrap__(self, out_arr, context)

    def to(self, device: str):
        print(str)
        
