from .errors import InvalidArguementError
def _ntuple(i, n):
    def parse(x):
        if type(x) is int:
            return [x for _ in range(n)]
        else:            
            return [1 for _ in range(n-i)] + [x[j] for j in range(i)]
    return parse

single = _ntuple(1, 3)
double = _ntuple(2, 3)
triple = _ntuple(3, 3)

def _reverse_repeat_tuple(t, n):
    return tuple(x for x in reversed(t) for _ in range(n))