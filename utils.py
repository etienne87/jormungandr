from __future__ import print_function
import pickle
import numpy as np
import torch

class Bunch:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)


def save_model(model, filename):
    state = {
        'net': model.state_dict(),
        'netobject': pickle.dumps(model),
    }
    torch.save(state, filename)


def load_model(filename, model=None):
    checkpoint = torch.load(filename)
    if model is None:
        model = pickle.load(checkpoint['netobject'])
    model.load_state_dict(checkpoint['net'])
    return model


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