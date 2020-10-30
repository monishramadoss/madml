from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import backend
from typing import List, Union
import numpy as np
import struct 

class tensor(backend.tensor):
    host_side = List[float]

    def __init__(self, data: Union[np.ndarray, List[Union[float, int]]], shape: List[int]) -> None:
        self.host_side = data

        if type(data) == np.ndarray:
            self.host_side = data.reshape(-1).tolist()

        if type(self.host_side[0]) == int:
            for i in range(len(data)):
                self.host_side[i] = float(self.host_side[i])
        super(tensor, self).__init__(self.host_side, shape)

    def __len__(self) -> int:
        return self.size()

    def __getitem__(self, idx: int):
        assert(self.size > idx)
        self.backend_layer = None
        new_shape = self.shape[1:]
        new_size = self._size(new_shape)
        new_data = list()

        #_data = self._convert_to_float(self.byte_size,self.tnsr.toHost())
        #for i in range(new_size):
        #    new_data.append(self._data[idx * new_size + i])
        #return tensor(new_data, new_shape)

    def T(self):
        return self.reshape([self.shape[1], self.shape[0]])

    def _convert_to_float(self, size:int, arr:List[bytes]) -> List[float]:
        ret_data = []
        ret_data.extend([bytearray(arr[i:i+4]) for i in range(0, size, 4)])
        for i in range(len(ret_data)):
            ret_data[i] = struct.unpack("f", ret_data[i])
        return ret_data

    def numpy(self):
        return np.array(self.host_side).reshape(self.shape)
