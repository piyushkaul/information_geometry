
# Code taken from Notes and assignments for Stanford CS class CS231n: Convolutional Neural Networks for Visual Recognition
# https://github.com/cs231n/cs231n.github.io
# The following license is to be included as per license terms.
# The MIT License (MIT)
#
# Copyright (c) 2015 Andrej Karpathy
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
cimport numpy as np
cimport cython

# DTYPE = np.float64
# ctypedef np.float64_t DTYPE_t

ctypedef fused DTYPE_t:
    np.float32_t
    np.float64_t

def im2col_cython(np.ndarray[DTYPE_t, ndim=4] x, int field_height,
                  int field_width, int padding, int stride):
    cdef int N = x.shape[0]
    cdef int C = x.shape[1]
    cdef int H = x.shape[2]
    cdef int W = x.shape[3]
    
    cdef int HH = (H + 2 * padding - field_height) / stride + 1
    cdef int WW = (W + 2 * padding - field_width) / stride + 1

    cdef int p = padding
    cdef np.ndarray[DTYPE_t, ndim=4] x_padded = np.pad(x,
            ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    cdef np.ndarray[DTYPE_t, ndim=2] cols = np.zeros(
            (C * field_height * field_width, N * HH * WW),
            dtype=x.dtype)

    # Moving the inner loop to a C function with no bounds checking works, but does
    # not seem to help performance in any measurable way.

    im2col_cython_inner(cols, x_padded, N, C, H, W, HH, WW,
                        field_height, field_width, padding, stride)
    return cols


@cython.boundscheck(False)
cdef int im2col_cython_inner(np.ndarray[DTYPE_t, ndim=2] cols,
                             np.ndarray[DTYPE_t, ndim=4] x_padded,
                             int N, int C, int H, int W, int HH, int WW,
                             int field_height, int field_width, int padding, int stride) except? -1:
    cdef int c, ii, jj, row, yy, xx, i, col

    for c in range(C):
        for yy in range(HH):
            for xx in range(WW):
                for ii in range(field_height):
                    for jj in range(field_width):
                        row = c * field_width * field_height + ii * field_height + jj
                        for i in range(N):
                            col = yy * WW * N + xx * N + i
                            cols[row, col] = x_padded[i, c, stride * yy + ii, stride * xx + jj]



def col2im_cython(np.ndarray[DTYPE_t, ndim=2] cols, int N, int C, int H, int W,
                  int field_height, int field_width, int padding, int stride):
    cdef np.ndarray x = np.empty((N, C, H, W), dtype=cols.dtype)
    cdef int HH = (H + 2 * padding - field_height) / stride + 1
    cdef int WW = (W + 2 * padding - field_width) / stride + 1
    cdef np.ndarray[DTYPE_t, ndim=4] x_padded = np.zeros((N, C, H + 2 * padding, W + 2 * padding),
                                        dtype=cols.dtype)

    # Moving the inner loop to a C-function with no bounds checking improves
    # performance quite a bit for col2im.
    col2im_cython_inner(cols, x_padded, N, C, H, W, HH, WW, 
                        field_height, field_width, padding, stride)
    if padding > 0:
        return x_padded[:, :, padding:-padding, padding:-padding]
    return x_padded


@cython.boundscheck(False)
cdef int col2im_cython_inner(np.ndarray[DTYPE_t, ndim=2] cols,
                             np.ndarray[DTYPE_t, ndim=4] x_padded,
                             int N, int C, int H, int W, int HH, int WW,
                             int field_height, int field_width, int padding, int stride) except? -1:
    cdef int c, ii, jj, row, yy, xx, i, col

    for c in range(C):
        for ii in range(field_height):
            for jj in range(field_width):
                row = c * field_width * field_height + ii * field_height + jj
                for yy in range(HH):
                    for xx in range(WW):
                        for i in range(N):
                            col = yy * WW * N + xx * N + i
                            x_padded[i, c, stride * yy + ii, stride * xx + jj] += cols[row, col]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef col2im_6d_cython_inner(np.ndarray[DTYPE_t, ndim=6] cols,
                            np.ndarray[DTYPE_t, ndim=4] x_padded,
                            int N, int C, int H, int W, int HH, int WW,
                            int out_h, int out_w, int pad, int stride):

    cdef int c, hh, ww, n, h, w
    for n in range(N):
        for c in range(C):
            for hh in range(HH):
                for ww in range(WW):
                    for h in range(out_h):
                        for w in range(out_w):
                            x_padded[n, c, stride * h + hh, stride * w + ww] += cols[c, hh, ww, n, h, w]
    

def col2im_6d_cython(np.ndarray[DTYPE_t, ndim=6] cols, int N, int C, int H, int W,
        int HH, int WW, int pad, int stride):
    cdef np.ndarray x = np.empty((N, C, H, W), dtype=cols.dtype)
    cdef int out_h = (H + 2 * pad - HH) / stride + 1
    cdef int out_w = (W + 2 * pad - WW) / stride + 1
    cdef np.ndarray[DTYPE_t, ndim=4] x_padded = np.zeros((N, C, H + 2 * pad, W + 2 * pad),
                                                  dtype=cols.dtype)

    col2im_6d_cython_inner(cols, x_padded, N, C, H, W, HH, WW, out_h, out_w, pad, stride)

    if pad > 0:
        return x_padded[:, :, pad:-pad, pad:-pad]
    return x_padded 
