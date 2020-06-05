import numpy as np


class Tensor:
    self.shape = []
    self.is_gpu = False
    self.data = []
   
    def __init__(self, data):
        self.data = np.array(data)
        self.shape = self.data.shape

    def backward(self, gradient=None, retain_graph=None, create_graph=False):
        pass #autograd 

    def __len__(self):
        return self.shape[0]