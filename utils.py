from __future__ import print_function
import os
import csv
import pickle
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import cv2

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

def cv_can_load_it(filename):
    try:
        im = cv2.imread(filename)
        return True
    except:
        return False

# some files are 0bytes...
def can_load_it(filename):
    try:
        im = Image.open(filename)
        return True
    except:
        return False

def get_img_size(filename):
    try:
        im = Image.open(filename)
        return im.size
    except:
        return None


def subselect(files):
    good = []
    for file in files:
        if can_load_it(file):
            good.append(file)
    return good

def split_dataset(directory, train_out, val_out, all_out, ratio=0.7):
    """
    will create 2 csv (train/ val)
    :param path:
    :return:
    """
    train_dic, val_dic, all_dic = {}, {}, {}
    rank = 0

    subdirs = sorted([x[0] for x in os.walk(directory) if x[0] !=directory])

    for dir in subdirs:
        files = glob.glob(dir + '/*.jpg')
        files = subselect(files)
        print(os.path.basename(dir), ': ', len(files))
        idx = range(len(files))
        random.shuffle(idx)
        cut = int(ratio*len(files))

        train_files = [files[i] for i in idx[:cut]]
        val_files = [files[i] for i in idx[cut:]]

        # stupid check
        if len(set(train_files).intersection(val_files)) > 0:
            assert 0

        class_name = os.path.basename(dir)
        train_dic[class_name] = train_files
        val_dic[class_name] = val_files
        all_dic[class_name] = files

    pickle.dump(train_dic, open(train_out, "wb"))
    pickle.dump(val_dic, open(val_out, "wb"))
    pickle.dump(val_dic, open(all_out, "wb"))


def get_file_sizes(directory):
    subdirs = sorted([x[0] for x in os.walk(directory) if x[0] != directory])
    files = []
    for dir in subdirs:
        files += glob.glob(dir + '/*.jpg')
    sizes = []
    for file in files:
        size = get_img_size(file)
        if size is not None:
            sizes.append( size )
    return sizes

def plot_file_sizes(directory):
    sizes = get_file_sizes(directory)
    areas, ratios = [], []
    areas = [item[0]*item[1]*1e-3 for item in sizes]
    ratios = [float(item[0])/item[1] for item in sizes]

    ax1 = plt.subplot(221)
    ax2 = plt.subplot(222)
    ax1.hist(areas, label='areas')
    ax2.hist(ratios, label='ratios')
    plt.show()


def folder_to_pkl(dir, test_out):
    files = glob.glob(dir + '/*.jpg')
    dic = {}
    dic['class-0'] = files
    pickle.dump(dic, open(test_out, "wb"))


def merge_pickle(pkl1, pkl2, outfilename):
    dic1 = pickle.load(open(pkl1, 'r'))
    dic2 = pickle.load(open(pkl2, 'r'))
    dic = dic1.copy()
    dic.update(dic2)
    pickle.dump(dic, open(outfilename, "wb"))


def get_species(mapping_filename):
    species_to_idx = {}
    with open(mapping_filename, 'rb') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',', quotechar='|')
        for row in reader:
            species_to_idx[row['original_class']] = row['class_idx']
    return species_to_idx


def get_map(directory, mapping_filename):
    #1. get dict species -> class_id
    idx_to_species = {}
    with open(mapping_filename, 'rb') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',', quotechar='|')
        for row in reader:
            idx_to_species[int(row['class_idx'])] = row['original_class']
    #2. get contiguous_id -> specie
    nn_idx_to_species = {}
    subdirs = [x[0] for x in os.walk(directory) if x[0] != directory]
    subdirs = sorted(subdirs, key=lambda x: int(x.split('class-')[1]))
    for c, subdir in enumerate(subdirs):
        class_id = int(subdir.split('class-')[1])
        specie = idx_to_species[class_id]
        nn_idx_to_species[c] = specie

    assert len(idx_to_species) == len(nn_idx_to_species)
    return nn_idx_to_species