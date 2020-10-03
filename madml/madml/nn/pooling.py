from .module import Module
from madml.utils import single, double, triple

class _MaxPoolNd(Module):
    __constants__ = ['kernel_size', 'stride', 'padding', 'dilation', 'return_indices', 'ceil_mode']

    return_indices : bool
    ceil_mode : bool

    def __init__(self, kernel_size: Union[int, list[int]], stride: Optional[Union[int, list[int]]]=None,
                 padding: Union[int, list[int]]=0, dilation: Union[int, list[int]]=1,
                 return_indices: bool=False, ceil_mode: bool=False) -> None:
        super(_MaxPoolNd, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def extra_repr(self) -> str:
        return 'kernel_size={kernel_size}, stride={stride}, padding={padding}, dilation={dilation}, ceil_mode={ceil_mode}'.format(**self.__dict__)

class MaxPool1d(_MaxPoolNd):
    kernel_size : int
    stride : int
    padding : int
    dilation : int

class MaxPool2d(_MaxPoolNd):
    kernel_size : Union[int, list[int]]
    stride : Union[int, list[int]]
    padding : Union[int, list[int]]
    dilation : Union[int, list[int]]

class MaxPool3d(_MaxPoolNd):
    kernel_size : Union[int, list[int]]
    stride : Union[int, list[int]]
    padding : Union[int, list[int]]
    dilation : Union[int, list[int]]

class _MaxUnpoolNd(Module):
    __constants__ = ['kernel_size', 'stride', 'padding']

    kernel_size : Union[int, list[int]]
    stride : Optional[Union[int, list[int]]]
    padding : Union[int, list[int]]

    def __init__(self, kernel_size, stride, padding):
        super(_MaxUnpoolNd, self).__init__()
        self.kernel_size = kernel_size
        if stride:
            self.stride = stride
        self.padding = padding

    def extra_repr(self) -> str:
        return 'kernel_size={}, stride={}, padding={}'.format(self.kernel_size, self.stride, self.padding)

class MaxUnpool1d(_MaxUnpoolNd):
    def __init__(self, kernel_size: int, stride: Optional[int]=None, padding: int=0) -> None:
        kernel_size = single(kernel_size)
        padding = single(padding)
        if stride:
            stride = single(stride or kernel_size)
        super(MaxUnpool1d, self).__init__(kernel_size, stride, padding)

class MaxUnpool2d(_MaxUnpoolNd):
    def __init__(self, kernel_size: Union[int, list[int]], stride: Optional[Union[int, list[int]]]=None, padding: Union[int, list[int]]=0) -> None:
        kernel_size = double(kernel_size)
        padding = double(padding)
        if stride:
            stride = double(stride or kernel_size)
        super(MaxUnpool2d, self).__init__(kernel_size, stride, padding)

class MaxUnpool3d(_MaxUnpoolNd):
    def __init__(self, kernel_size: Union[int, list[int]], stride: Optional[Union[int, list[int]]]=None, padding: Union[int, list[int]]=0) -> None:
        kernel_size = triple(kernel_size)
        padding = triple(padding)
        if stride:
            stride = triple(stride or kernel_size)
        super(MaxUnpool3d, self).__init__(kernel_size, stride, padding)

class _AvgPoolNd(Module):
    __constants__ = ['kernel_size', 'stride', 'padding', 'ceil_mode', 'count_include_pad', 'divisor_override']
    kernel_size : Union[int, list[int]]
    stride : Union[int, list[int]]
    padding : Union[int, list[int]]
    ceil_mode : bool
    count_include_pad : bool
    divisor_override : bool

    def __init__(self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override) -> None:
        super(AvgPoolNd, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override

    def extra_repr(self) -> str:
        return 'kernel_size={}, stride={}, padding={}'.format(self.kernel_size, self.stride, self.padding)

class AvgPool1d(_AvgPoolNd):
    def __init__(self, kernel_size: int, stride: int=None, padding: int=0, ceil_mode: bool=False, count_include_pad: bool=True) -> None:
        kernel_size = single(kernel_size)
        stride = single(stride if stride is not None else kernel_size)
        padding = single(padding)
        super(AvgPool1d, self).__init__(kernel_size, stride, padding, ceil_mode, count_include_pad, False)

class AvgPool2d(_AvgPoolNd):
    def __init__(self, kernel_size: Union[int, list[int]], stride: Optional[Union[int, list[int]]]=None, padding: Union[int, list[int]]=0,
                 ceil_mode: bool=False, count_include_pad: bool=True, divisor_override: bool=None) -> None:
        kernel_size = double(kernel_size)
        stride = double(stride if stride is not None else kernel_size)
        padding = double(padding)
        super(AvgPool2d, self).__init__(kernel_size, stride, padding, ceil_mode, count_include_pad,)

    def __setstate__(self, d):
        super(AvgPool2d, self).__setstate__(d)
        self.__dict__.setdefault('padding', 0)
        self.__dict__.setdefault('ceil_mode', False)
        self.__dict__.setdefault('count_include_pad', True)

class AvgPool3d(_AvgPoolNd):
    def __init__(self, kernel_size: Union[int, list[int]], stride: Optional[Union[int, list[int]]]=None, padding: Union[int, list[int]]=0,
                 ceil_mode: bool=False, count_include_pad: bool=True, divisor_override: bool=None) -> None:
        kernel_size = triple(kernel_size)
        stride = triple(stride if stride is not None else kernel_size)
        padding = triple(padding)
        super(AvgPool2d, self).__init__(kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)

    def __setstate__(self, d):
        super(AvgPool3d, self).__setstate__(d)
        self.__dict__.setdefault('padding', 0)
        self.__dict__.setdefault('ceil_mode', False)
        self.__dict__.setdefault('count_include_pad', True)