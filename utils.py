from __future__ import print_function
import numpy as np

def make_grid(im, thumbsize=80):
    im2 = im.reshape(im.shape[0] / thumbsize, thumbsize, im.shape[1] / thumbsize, thumbsize, 3)
    im2 = im2.swapaxes(1, 2).reshape(-1, thumbsize, thumbsize, 3)
    return im2


def unmake_grid(batch):
    batch = np.transpose(batch, [0, 2, 3, 1])
    batchsize = batch.shape[0]
    thumbsize = batch.shape[1]
    channels = batch.shape[-1]
    nrows = 2 ** ((batchsize.bit_length() - 1) // 2)
    ncols = batchsize / nrows
    im = batch.reshape(nrows, ncols, thumbsize, thumbsize, channels)
    im = im.swapaxes(1, 2)
    im = im.reshape(nrows * thumbsize, ncols * thumbsize, channels)
    return im