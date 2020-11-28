import numpy as np
import math
'''
https://github.com/pytorch/pytorch/blob/0f675f9cbc71f59eaff9930a6ee3c62176b18017/aten/src/ATen/native/Im2Col.cpp
https://github.com/pytorch/pytorch/blob/0f675f9cbc71f59eaff9930a6ee3c62176b18017/aten/src/ATen/native/im2col.h
https://github.com/pytorch/pytorch/blob/b90fc52c687a6851047f18ec9d06fb998efe99dd/aten/src/ATen/native/tensorProperties.cpp
'''

def im2col(A, kernel, pad, stride, dilation):
    batch_size, channels = A.shape[0], A.shape[1]
    height, width = A.shape[-2], A.shape[-1]
    A = A.flatten()
    pad_h, pad_w = pad
    kernel_h, kernel_w = kernel
    stride_h, stride_w = stride[0], stride[1]
    dilation_h, dilation_w = dilation

    height_col = int((height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) // stride_h + 1)
    width_col = int((width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) // stride_w + 1)

    n_output_plane = int(channels * kernel_w * kernel_h)
    output_length = int(batch_size * height_col * width_col)
    B = np.zeros(shape=int(batch_size * n_output_plane * output_length))

    for elt in range(batch_size):
        data_im = elt * channels * height * width
        data_col = elt * n_output_plane * output_length

        for index in range(int(channels * height_col * width_col)):
            w_offset = int(index % kernel_w)
            h_offset = int((index / kernel_w) % kernel_h)
            c_im = int(index / kernel_h / kernel_w)
            for h_col in range(int(height_col)):
                h_im = int(h_col * stride_h - pad_h + h_offset * dilation_h)
                for w_col in range(int(width_col)):
                    w_im = int(w_col * stride_w - pad_w + w_offset * dilation_w)
                    if h_im >= 0 and h_im < height and w_im >= 0 and w_im < width:
                        col_idx = int(data_col + (index * height_col + h_col) * width_col + w_col)
                        im_idx = int(data_im + (c_im * height + h_im) * width + w_im)
                        if im_idx < A.shape[0]:
                            B[col_idx] += A[im_idx]

    return B.reshape(n_output_plane, output_length)

def col2im(A, kernel, pad, stride, dilation):
    batch_size, channels = A.shape[0], A.shape[1]
    height, width = A.shape[-2], A.shape[-1]
    A = A.flatten()
    pad_h, pad_w = pad
    kernel_h, kernel_w = kernel
    stride_h, stride_w = stride
    dilation_h, dilation_w = dilation

    height_col = int((height - 1) * stride_h - 2 * pad_h + dilation_h * (kernel_h - 1) + 1)
    width_col = int((width - 1) * stride_w - 2 * pad_w + dilation_w * (kernel_w - 1) + 1)
    n_output_plane = int(channels * kernel_w * kernel_h)
    output_length = int(batch_size * height_col * width_col)
    B = np.zeros(shape=int(batch_size * n_output_plane * output_length))

    for elt in range(batch_size):
        data_im = elt * channels * height * width
        data_col = elt * n_output_plane * output_length

        for index in range(int(channels * height_col * width_col)):
            w_offset = int(index % kernel_w)
            h_offset = int((index / kernel_w) % kernel_h)
            c_im = int(index / kernel_h / kernel_w)
            for h_col in range(int(height_col)):
                h_im = int(h_col * stride_h - pad_h + h_offset * dilation_h)
                for w_col in range(int(width_col)):
                    w_im = int(w_col * stride_w - pad_w + w_offset * dilation_w)
                    if h_im >= 0 and h_im < height and w_im >= 0 and w_im < width:
                        col_idx = int(data_col + (index * height_col + h_col) * width_col + w_col)
                        im_idx = int(data_im + (c_im * height + h_im) * width + w_im)
                        if im_idx < A.shape[0]:
                            B[col_idx] = A[im_idx]
    return B.reshape(n_output_plane, output_length)

def im2col_2(A, kernel, pad, stride, dilation, padding_type='zero'):
    A = np.transpose(A, axes=(1,0,2,3))
    batch_size, channels = A.shape[0], A.shape[1]
    height, width = A.shape[-2], A.shape[-1]
    pad_h, pad_w = pad
    kernel_h, kernel_w = kernel
    stride_h, stride_w = stride
    dilation_h, dilation_w = dilation

    if padding_type == 'zero':
        pad = ((kernel[0] - 1) * dilation_h - pad[0], (kernel[1] - 1) * dilation_w - pad[1])
    if padding_type == 'same':
        pad = (math.floor(kernel_h / 2), math.floor(kernel_w / 2))
        kernel = (pad[0] * 2 + 1, pad[1] * 2 + 1)
    if padding_type == 'full':
        kernel = (kernel_h - 1, kernel_w - 1)
        pad = kernel
    if stride[0] > 0 and stride[1] > 0:
        stride = (1 / (stride[0]), 1 / (stride[1]))

    y = im2col(A, kernel, pad, stride, dilation)
    print(y.shape, '\n')
    return y

#     for elt in range(batch_size):
#     data_im = elt * channels * height * width
#     data_col = elt * channels * height_col * width_col
#     for index in range(channels * kernel_h * kernel_w):
#         w_offset = int(index % kernel_w)
#         h_offset = int((index / kernel_w) % kernel_h)
#         c_im = int(index / kernel_h / kernel_w)
#         for h_col in range(int(height_col)):
#             h_im = h_col * stride_h + pad_h + h_offset * dilation_h
#             for w_col in range(int(width_col)):
#                 w_im = w_col * stride_w + pad_w + w_offset * dilation_w
#                 if h_im >= 0 and h_im < height and w_im >= 0 and w_im <
#                 width:
#                     im_idx = int(data_im + (c_im * height + h_im) * width +
#                     w_im)
#                     col_idx = int(data_col + (index * height_col + h_col) *
#                     width_col + w_col)
#                     B[im_idx] += A[col_idx]
BATCH_SIZE = 1
if __name__ == "__main__":
    inpt = np.arange(0, 25, dtype=np.float32).reshape(1, 1, 5, 5)
    weight = np.ones((1,3,3)).reshape(1,-1)
    ic = im2col(inpt, (3,3), (1,1), (1,1), (1,1)) # 9x25
    print(ic)
    # ot = np.matmul(weight, ic)
    # x = np.array([[[[0., 1., 2.], # (1, 1, 3, 3)
    #             [3., 4., 5.],
    #             [6., 7., 8.]]]]).astype(np.float32)

    # W = np.array([[[[1., 1., 1.], # (1, 2, 3, 3)
    #             [1., 1., 1.],
    #             [1., 1., 1.]],
    #            [[1., 1., 1.],
    #             [1., 1., 1.],
    #             [1., 1., 1.]]]]).astype(np.float32)
    # W = W.reshape(2, -1)
    # ic = im2col_2(x, kernel=(3,3), pad=(0,0), stride=(1,1), dilation=(1,1))
    # ot = np.matmul(W, ic).reshape(-1, 2, 5, 5)
    # print(ot)
    # print(ot.shape, '\n')
    x = np.arange(0, 25, dtype=np.float32).reshape(1,1,5,5)
    y = np.transpose(x, (0,1,3,2))
    x2 = np.transpose(y, (0,1,3,2))
    print(x == x2)
    # x = np.arange(0, 25, dtype=np.float32).reshape(1, 1, 5, 5)

    # W = np.array([[[[7., 2.],  # (1, 1, 2, 2)
    #                 [1., 9.]]]]).astype(np.float32)

    # W = W.reshape(1, -1)
    # ic = im2col_2(x, kernel=(3,3), pad=(1,1), stride=(2,2), dilation=(1,1))