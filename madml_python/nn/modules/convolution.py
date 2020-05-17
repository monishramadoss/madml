import numpy as np
'''
https://github.com/pytorch/pytorch/blob/0f675f9cbc71f59eaff9930a6ee3c62176b18017/aten/src/ATen/native/Im2Col.cpp
https://github.com/pytorch/pytorch/blob/0f675f9cbc71f59eaff9930a6ee3c62176b18017/aten/src/ATen/native/im2col.h
https://github.com/pytorch/pytorch/blob/b90fc52c687a6851047f18ec9d06fb998efe99dd/aten/src/ATen/native/TensorProperties.cpp
'''

def im2col(A, channels, height, width, kernel, pad, stride, dilation):
    A = A.flatten()
    pad_h, pad_w = pad
    kernel_h, kernel_w = kernel
    stride_h, stride_w = stride
    dilation_h, dilation_w = dilation

    height_col = int((height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1)
    width_col = int((width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1)

    batch_size = inpt.shape[0]

    n_output_plane = int(channels * kernel_w * kernel_h)
    output_length = int(height_col * width_col)

    B = np.zeros(shape=int(batch_size * n_output_plane * output_length))

    for elt in range(batch_size):
        data_im, data_col, = elt * channels * height * width, elt * n_output_plane * output_length

        for index in range(int(channels * height_col * width_col)):
            w_offset = int(index % kernel_w)
            h_offset = int((index / kernel_w) % kernel_h)
            c_im =int(index / kernel_h / kernel_w)

            for h_col in range(int(height_col)):
                h_im = int(h_col * stride_h - pad_h + h_offset * dilation_h)
                for w_col in range(int(width_col)):
                    w_im = int(w_col * stride_w - pad_w + w_offset * dilation_w)
                    if h_im >= 0 and  h_im < height and w_im >= 0 and w_im < width:
                        col_idx = int(data_col + (index * height_col + h_col) * width_col + w_col)
                        im_idx = int(data_im + (c_im * height + h_im) * width + w_im)
                        if im_idx < A.shape[0]:
                            B[col_idx] = A[im_idx]
                    else:
                        B[int(data_col + (index * height_col + h_col) * width_col + w_col)] = 0


    return B.reshape((batch_size, n_output_plane, output_length))


def col2im(A, channels, height, width, kernel, pad, stride, dilation):
    pad_h, pad_w = pad
    kernel_h, kernel_w = kernel
    stride_h, stride_w = stride
    dilation_h, dilation_w = dilation

    height_col = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1
    width_col = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1
    kernel_extent_w = (kernel_w - 1) * dilation_w + 1
    kernel_extent_h = (kernel_h - 1) * dilation_h + 1

    batch_size = A.shape[0]
    n_input_plane = A.shape[1]
    n_output_plane = int(n_input_plane / (kernel_w * kernel_h))

    A = A.flatten()
    B = np.zeros(shape=int(batch_size * n_output_plane * height * width))

    for elt in range(batch_size):
        data_im, data_col, = elt * channels * height * width, elt * channels * height_col * width_col
        for index in range(channels * kernel_h * kernel_w):
            w_offset = int(index % kernel_w)
            h_offset = int((index / kernel_w) % kernel_h)
            c_im = int(index / kernel_h / kernel_w)
            for h_col in range(int(height_col)):
                h_im = h_col * stride_h + pad_h + h_offset * dilation_h
                for w_col in range(int(width_col)):
                    w_im = w_col * stride_w + pad_w + w_offset * dilation_w
                    if h_im >= 0 and h_im < height and w_im >= 0 and w_im < width:
                        im_idx = int(data_im + (c_im * height + h_im) * width + w_im)
                        col_idx = int(data_col + (index * height_col + h_col) * width_col + w_col)
                        B[im_idx] += A[col_idx]

    return B.reshape((batch_size, n_output_plane, height, width))

BATCH_SIZE = 1
if __name__ == "__main__":
    inpt = np.ones(shape=(BATCH_SIZE, 3, 128, 128))
    output = im2col(inpt, 3, 128, 128, (3,3), (0,0), (1,1), (1,1))
    print(output.shape, output[output==1].shape, output[output==0].shape)
    tmp = col2im(output, 3, 128, 128, (3,3), (0,0), (1,1), (1,1))
    print(tmp.shape, tmp[tmp==1].shape, tmp[tmp==0].shape)