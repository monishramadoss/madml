class Module:
    def __init__(self,):
        pass

    def __forward__(self, x):
        return self.__call__(x)

    def __call__(self, *args):
        pass