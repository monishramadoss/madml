import numpy as np


class Tensor:
    self.shape = []
    self.is_gpu = False
    self.data = []
    

    def backward(self, gradient=None, retain_graph=None, create_graph=False):
        pass #autograd 

    def __len__(self):
        return self.shape[0]