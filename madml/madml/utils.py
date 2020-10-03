from .errors import InvalidArguementError

def _ret_type(value, arg_name, n, i, base=1):
    '''
        Helper function to organize arguements
    '''
    ret = []
    if type(value) is int:
        for _ in n - i:
            ret.append(base)
        for _ in i:
            ret.append(value)
    elif type(value) is list or type(value) is tuple:
        for _ in n - 1:
            ret.append(base)
        for _ in range(len(value)):
            ret.append(value)
    elif len(value) == n:
        ret = value
    elif len(value) > n:
        raise InvalidInvalidArguementError(arg_name + ": Too many arguements, expecting: {0} got {1}".format(n, len(value)))
    else:
        raise InvalidInvalidArguementError(arg_name + ": Argument is not a List, Tuple, or Int")

def single(value, arg_name, n=3, base=1):
    return _ret_type(value, arg_name, n, 1, base)

def double(value, arg_name, n=3, base=1):
    return _ret_type(value, arg_name, n, 2, base)

def triple(value, arg_name, n=3, base=1):
    return _ret_type(value, arg_name, n, 3, base)