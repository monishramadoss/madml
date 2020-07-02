
import numpy as np
#from .module import Module


'''
https://github.com/pytorch/pytorch/blob/0f675f9cbc71f59eaff9930a6ee3c62176b18017/aten/src/ATen/native/vol2col.cpp
https://github.com/pytorch/pytorch/blob/0f675f9cbc71f59eaff9930a6ee3c62176b18017/aten/src/ATen/native/vol2col.h
https://github.com/pytorch/pytorch/blob/b90fc52c687a6851047f18ec9d06fb998efe99dd/aten/src/ATen/native/TensorProperties.cpp
'''
output_h = 0
output_w = 0
def vol2col(A, channels, height, width, kernel, pad, stride, dilation):
    A = A.flatten()
    pad_h, pad_w = pad
    kernel_h, kernel_w = kernel
    stride_h, stride_w = stride
    dilation_h, dilation_w = dilation

    height_col = int((height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1)
    width_col = int((width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1)
    global output_h
    global output_w
    output_h = height_col
    output_w = width_col
    print("out:", height_col, width_col, "in:", height, width,)
    batch_size = inpt.shape[0]

    n_output_plane = int(channels * kernel_w * kernel_h)
    output_length = int(height_col * width_col)

    B = np.zeros(shape=int(batch_size * n_output_plane * output_length))
    wrong_counter = 0
    channels_col = channels * kernel_h * kernel_w

    for elt in range(batch_size):
        data_vol = elt * channels * height * width
        data_col = elt * n_output_plane * output_length
        #data_vol, data_col, = 0, 0
        for c_col in range(channels_col):
            w_offset = int(c_col % kernel_w)
            h_offset = int((c_col / kernel_w) % kernel_h)
            c_vol = int(c_col / kernel_h / kernel_w)

            for h_col in range(int(height_col)):
                h_vol = int(h_col * stride_h - pad_h + h_offset * dilation_h)
                for w_col in range(int(width_col)):
                    w_vol = int(w_col * stride_w - pad_w + w_offset * dilation_w)
                    if 0 <= h_vol < height and 0 <= w_vol < width:
                        col_idx = int(data_col + (c_col * height_col + h_col) * width_col + w_col)
                        vol_idx = int(data_vol + (c_vol * height + h_vol) * width + w_vol)
                        B[col_idx] = A[vol_idx]
                    else:
                        B[int(data_col + (c_col * height_col + h_col) * width_col + w_col)] = 0

    print(wrong_counter)
    return B.reshape((batch_size, n_output_plane, output_length))


def col2vol(A, channels, height, width, kernel, pad, stride, dilation):
    pad_h, pad_w = pad
    kernel_h, kernel_w = kernel
    stride_h, stride_w = stride
    dilation_h, dilation_w = dilation
	#(X[0]−1) × stride[0]−2 × padding[0] + dilation[0] × (kernel_size[0]−1)+output_padding[0]+1

    height_col = height #(height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1
    width_col = width #(width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1
    kernel_extent_w = (kernel_w - 1) * dilation_w + 1
    kernel_extent_h = (kernel_h - 1) * dilation_h + 1
    height  = (height - 1) * stride_h -2 * pad_h + dilation_h * (kernel_h - 1) + pad_h + 1
    width 	= (width - 1) * stride_w - 2 * pad_w + dilation_w * (kernel_w - 1) + pad_w + 1
    print("out:", height, width, "in:", height_col, width_col)
   
    batch_size = A.shape[0]
    n_input_plane = A.shape[1]
    print(n_input_plane)
    n_output_plane = int(n_input_plane / (kernel_w * kernel_h))

    A = A.flatten()
    B = np.zeros(shape=int(batch_size * n_output_plane * height * width))
    channels_col = channels * kernel_h * kernel_w
    print(A.shape, B.shape)
 
    for elt in range(batch_size):
        data_col = elt * n_output_plane * height * width
        data_vol = elt * channels * height_col * width_col
        for index in range(channels_col):
            w_offset = int(index % kernel_w)
            h_offset = int((index / kernel_w) % kernel_h)
            c_vol = int(index / kernel_h / kernel_w)
            for h_col in range(int(height_col)):
                h_vol = h_col * stride_h - pad_h + h_offset * dilation_h
                for w_col in range(int(width_col)):
                    w_vol = w_col * stride_w - pad_w + w_offset * dilation_w
                    if 0 <= h_vol < height and 0 <= w_vol < width:
                        vol_idx = int(data_vol + (c_vol * height + h_vol) * width + w_vol)
                        col_idx = int(data_col + (index * height_col + h_col) * width_col + w_col)
                        xtmp = A[col_idx]
                        B[vol_idx] += xtmp

    return B.reshape((batch_size, channels, height, width))

BATCH_SIZE = 2
Height = 5
Width = 5


kernel = np.ones(shape=(8,3,3,3)).reshape((8, 27))
if __name__ == "__main__":
    inpt = np.ones(shape=(BATCH_SIZE, 3, Height, Width))
    # vol2col(A, channels, height, width, kernel, pad, stride, dilation):
    output = vol2col(inpt, 3, Height, Width, (3,3), (0,0), (1,1), (1,1))
    #print(output)
    print("::", output.shape) #, output[output==1].shape, output[output==0].shape)
    print(kernel.shape)
    output = np.matmul(kernel, output[0])
    print(output.shape)

    tmp = col2vol(output, 8, 3, 3, (3,3), (0,0), (1,1), (1,1))
    print(tmp.shape) #, tmp[tmp!=0].shape)
    #print(tmp)





# class _ConvNd(Module):
#     __constants__ = ['stride', 'padding', 'dilation', 'groups',
#                      'padding_mode', 'output_padding', 'in_channels',
#                      'out_channels', 'kernel_size']

#     def __init__(self, in_channels, out_channels, kernel_size, stride,
#                  padding, dilation, transposed, output_padding,
#                  groups, bias, padding_mode):
#         super(_ConvNd, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.padding = padding
#         self.dilation = dilation
#         self.transposed = transposed
#         self.output_padding = output_padding
#         self.groups = groups
#         self.padding_mode = padding_mode
#     def forward(self, x):
#         pass

# class _ConvTransposeNd(_ConvNd):
#     def __init__(self, in_channels, out_channels, kernel_size, stride,
#                  padding, dilation, transposed, output_padding,
#                  groups, bias, padding_mode):
#         if padding_mode != 'zeros':
#             raise ValueError('Only "zeros" padding mode is supported for {}'.format(self.__class__.__name__))

#         super(_ConvTransposeNd, self).__init__(
#             in_channels, out_channels, kernel_size, stride,
#             padding, dilation, transposed, output_padding,
#             groups, bias, padding_mode)
    
#     def forward(self, x):
#         pass
#     