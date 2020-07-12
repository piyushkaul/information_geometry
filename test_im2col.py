import numpy as np
from im2col import im2col_indices

x = np.random.randn(5,10,20,30)
w = np.random.randn(32,10,3,3)

N, C, H, W = x.shape
num_filters, _, filter_height, filter_width = w.shape
stride, pad = 1, 1

# Check dimensions
assert (W + 2 * pad - filter_width) % stride == 0, 'width does not work'
assert (H + 2 * pad - filter_height) % stride == 0, 'height does not work'

# Create output
out_height = (H + 2 * pad - filter_height) // stride + 1
out_width = (W + 2 * pad - filter_width) // stride + 1
out = np.zeros((N, num_filters, out_height, out_width), dtype=x.dtype)
x_cols = im2col_indices(x, w.shape[2], w.shape[3], pad, stride)

#y = im2col_indices(x, int(3), int(3), padding=int(1), stride=int(1))

print('x = {}'.format(x))
print('y = {}'.format(x_cols))