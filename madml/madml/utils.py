from .errors import InvalidArguementError
def _ntuple(n):
    def parse(x):
        return tuple(x, n)
    return _ntuple

single = _ntuple(1)
double = _ntuple(2)
triple = _ntuple(3)

def _reverse_repeat_tuple(t, n):
    return tuple(x for x in reversed(t) for _ in range(n))