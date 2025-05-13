from ast import Str
from builtins import range
import numpy as np
from .layers import *


def conv_forward_naive(x, w, b, conv_param):
    """A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # *****START OF YOUR CODE *****

    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    stride = conv_param["stride"]
    pad = conv_param["pad"]

    H_out = 1 + (H + 2 * pad - HH) // stride
    W_out = 1 + (W + 2 * pad - WW) // stride

    out = np.zeros((N, F, H_out, W_out))

    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")

    for n in range(N):
        for f in range(F):
            for i in range(H_out):
                for j in range(W_out):
                    x_region = x_padded[
                        n, :, i * stride : i * stride + HH, j * stride : j * stride + WW
                    ]
                    out[n, f, i, j] = np.sum(x_region * w[f]) + b[f]
    x = x_padded
    # *****END OF YOUR CODE *****
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives, of shape (N, F, H', W')
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, C, H, W)
    - dw: Gradient with respect to w, of shape (F, C, HH, WW)
    - db: Gradient with respect to b, of shape (F,)
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # *****START OF YOUR CODE *****

    x, w, b, conv_param = cache
    N, F, H_out, W_out = dout.shape
    _, _, HH, WW = w.shape
    stride = conv_param["stride"]
    pad = conv_param["pad"]

    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)

    for n in range(N):
        for f in range(F):
            for i in range(H_out):
                for j in range(W_out):
                    x_region = x[
                        n, :, i * stride : i * stride + HH, j * stride : j * stride + WW
                    ]
                    dw[f] += dout[n, f, i, j] * x_region
                    db[f] += dout[n, f, i, j]
                    dx[
                        n, :, i * stride : i * stride + HH, j * stride : j * stride + WW
                    ] += w[f] * dout[n, f, i, j]

    dx = dx[:, :, pad:-pad, pad:-pad]

    # *****END OF YOUR CODE *****
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here, eg you can assume:
      - (H - pool_height) % stride == 0
      - (W - pool_width) % stride == 0

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    # *****START OF YOUR CODE *****

    N, C, H, W = x.shape
    pool_height = pool_param["pool_height"]
    pool_width = pool_param["pool_width"]
    stride = pool_param["stride"]

    H_out = 1 + (H - pool_height) // stride
    W_out = 1 + (W - pool_width) // stride

    out = np.zeros((N, C, H_out, W_out))

    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    x_region = x[
                        n,
                        c,
                        i * stride : i * stride + pool_height,
                        j * stride : j * stride + pool_width,
                    ]
                    out[n, c, i, j] = np.max(x_region)

    # *****END OF YOUR CODE *****
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H', W')
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x, of shape (N, C, H, W)
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # *****START OF YOUR CODE *****

    x, pool_param = cache
    N, C, H_out, W_out = dout.shape
    pool_height = pool_param["pool_height"]
    pool_width = pool_param["pool_width"]
    stride = pool_param["stride"]

    dx = np.zeros_like(x)

    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    x_region = x[
                        n,
                        c,
                        i * stride : i * stride + pool_height,
                        j * stride : j * stride + pool_width,
                    ]
                    max_idx = np.unravel_index(np.argmax(x_region), x_region.shape)
                    dx[n, c, i * stride + max_idx[0], j * stride + max_idx[1]] += dout[
                        n, c, i, j
                    ]

    # *****END OF YOUR CODE *****
    return dx
