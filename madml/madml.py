
class Model:
    def __init__ (self):
        self.layers = list()
        self.loss = None
        self.loss_prime = None
    def add(self, layer):
        self.layers.append(layer)
    def use(self, loss, loss_prime, opt):
        self.loss = loss
        self.loss_prime = loss_prime


    def fit(self, x_train, y_train):
        pass