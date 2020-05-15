import numpy as np


def im2col(A, B, data_im, data_col, channels, height, width, kernel, pad, stride, dilation):
    pad_h, pad_w = pad
    kernel_h, kernel_w = kernel
    stride_h, stride_w = stride
    dilation_h, dilation_w = dilation

    height_col = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1
    width_col = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1

    for index in range(channels * height_col * width_col):
        h_index = index / width_col
        h_col = h_index % height_col
        w_col = index % width_col
        c_im = h_index / height_col
        c_col = channels * kernel_h * kernel_w
        h_offset = h_col * stride_h - pad_h
        w_offset = w_col * stride_w - pad_w

        data_col_ptr = data_col
        data_col_ptr += (c_col * height_col + h_col) * width_col + w_col
        data_im_ptr = data_im
        data_im_ptr += (c_im * height + h_offset) * width + w_offset

        for i in range(kernel_h):
            for j in range(kernel_w):
                h_im = h_offset + j * dilation_h
                w_im = w_offset + j * dilation_w

                if h_im >= 0 and w_im >= 0 and h_im < height and w_im < width:
                    B[data_col_ptr] = A[data_im_ptr + i * dilation_h * width + j * dilation_w]
                else:
                    B[data_im_ptr] = 0
                data_col_ptr += height_col * width_col


def col2im(A, B, data_col, data_im, channels, height, width, kernel, pad, stride, dilation):
    pad_h, pad_w = pad
    kernel_h, kernel_w = kernel
    stride_h, stride_w = stride
    dilation_h, dilation_w = dilation

    height_col = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1
    width_col = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1
    kernel_extent_w = (kernel_w - 1) * dilation_w + 1
    kernel_extent_h = (kernel_h - 1) * dilation_h + 1

    for index in range(channels * height * width):
        val = 0
        w_im = index % width + pad_w
        h_im = (index / width) % height + pad_h
        c_im = index / (width * height)
        if w_im < kernel_extent_w:
            w_col_start = 0
        else:
            w_col_start = (w_im - kernel_extent_w) / stride_w + 1
        w_col_end = min(w_im / stride_w + 1, width_col)

        if h_im < kernel_extent_h:
            h_col_start = 0
        else:
            h_col_start = (h_im - kernel_extent_h) / stride_h + 1
        h_col_end = min(h_im / stride_h + 1, height_col)

        for h_col in range(h_col_start, h_col_end):
            for w_col in range(w_col_start, w_col_end):
                h_k = h_im - h_col * stride_h
                w_k = w_im - w_col * stride_w
                if h_k % dilation_h == 0 and w_k % dilation_w == 0:
                    h_k /= dilation_h
                    w_k /= dilation_w
                    data_col_index = (((c_im * kernel_h + h_k) * kernel_w + w_k) * height_col + h_col) * width_col + w_col
                    val += A[data_col + data_col_index]
        B[data_im + index] = val

