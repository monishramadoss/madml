import backend
import numpy as np
from typing import Union, List

class tensor:
    def __init__(self, data: Union[List[float], np.ndarray], shape: List[int]=None):
        self.data = np.asarray(data)
        if(shape is None):
            self.shape = [len(data)] if type(data) is list else data.shape
        self.size = self._size(self.shape)

    def reshape(self, shape: List[List]):
        new_size = self._size(shape)
        new_shape = shape
        if(new_size == self.size):
            self.shape = shape
        else:   
            t = []
            for i in range(len(shape)):
                if new_shape[i] == -1:
                    new_shape[i] = new_size // self.size
            self.shape = new_shape        

    def _size(self, shape):
        size = 1
        for s in shape:
            size *= s
        return size       
