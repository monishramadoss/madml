#import backend
import numpy as np

class tensor:
    def __init__(self, shape:list(), data=None, dtype=float, init=False):
        self.shape = shape
        if not init or data is not None:
            self.data = np.zeros(shape) if data is None else data
        else:
            self.data = data
    
    def reshape(self, shape):
        self.data.reshape(shape)

    
    def __eq__(self, value):
        return tensor(self.shape, data=self.data == value)

    
    def __ne__(self, value):
        return tensor(self.shape, data=self.data != value)

    
    def __le__(self, value):
        return tensor(self.shape, data=self.data < value)

    
    def __gt__(self, value):
        return tensor(self.shape, data=self.data > value)

    
    def __add__(self, value):
        return tensor(self.shape, data=self.data + value)

    
    def __sub__(self, value):
        return tensor(self.shape, data=self.data - value)

    
    def __mul__(self, value):
        return tensor(self.shape, data=self.data * value)

    
    def __div__(self, value):
        return tensor(self.shape, data=self.data * value)