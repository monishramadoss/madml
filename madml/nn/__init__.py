from .activation import relu, dropout, softmax
from .convolution import conv1d, conv2d, conv3d
from .linear import linear
from .loss import crossentropyloss, mseloss
from .math import add, sub, mul, div, sigmoid
from .module import Module, Parameter
from .pooling import maxpool1d, maxpool2d, maxpool3d
from .transform import transpose, flatten

# TODO BUG: dy is not being updated
# TODO backward does not work because backward_args are registerd before backward is called
# TODO needs an update mechanism to detect bakend changes
# TODO validate deallocations
# TODO work on static members
# TODO nees systems for bradcasting uni/binary operations
# TODO nneeds async functional hooks
# TODO 