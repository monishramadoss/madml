class Module:
    def __init__(self,):
        pass

    def forward(self, x):
        raise NotImplemented( "{} forward for layer not Implemented".format(self.__name__))    

    def __call__(self, *args):
        return self.forward(args)