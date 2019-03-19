from __future__ import print_function
import os
import sys
import glob
import random
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader
import data_augmentation as da
import cv2


class ResizeCV(object):
    def __init__(self, imsize, keep_ratio=True):
        self.imsize = imsize
        self.keep_ratio = keep_ratio

    def keep_ratio_resize(self, sample):
        h, w, c = sample.shape
        h2, w2 = self.imsize

        if w > h:
            w3 = w2
            h3 = w2 * h / w
        else:
            h3 = h2
            w3 = h2 * w / h

        tmp = np.zeros((h2, w2, 3), dtype=np.uint8)

    def __call__(self, sample):
        #respect of aspect ratio

        image = cv2.resize(sample, self.imsize, 0, 0, interpolation=cv2.INTER_AREA)
        return image


class RandomDA(object):
    def __call__(self, sample):
        image = sample
        height, width, c = image.shape
        h = da.get_random_homography(height=height, width=width, perspective_range=0.00001)
        image = da.cv2_apply_transform_image(image, h)
        return image


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __getitem__(self, i):
        return self.transforms[i]

    def __call__(self, x):
        for transform in self.transforms:
            x = transform(x)
        return x


class SnakeDataset(Dataset):
    """Snake Facebook."""

    def __init__(self, pkl_file, transform=None):
        self.dic = pickle.load(open(pkl_file, 'r'))
        assert self.dic is not None
        self.get_list(self.dic)
        self.transform = transform

    def get_list(self, dic):
        samples = []
        skeys = sorted(dic.keys(), key=lambda x:int(x.split('class-')[1]))
        for c, key in enumerate(skeys):
            for file in dic[key]:
                samples.append((c,file))

        idx = range(len(samples))
        random.shuffle(idx)
        self.samples = [samples[i] for i in idx]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        label, img_name = self.samples[idx]
        image = cv2.imread(img_name)
        if image is None:
            print('Could not read: ', img_name)
            assert 0
        if self.transform is not None:
            image = self.transform(image)
        return (image, label, img_name)


if __name__ == '__main__':
    from torchvision import transforms
    from utils import unmake_grid

    # train_path = os.path.join(sys.argv[1], "train.pkl")
    # val_path = os.path.join(sys.argv[1], "val.pkl")
    # all_path = os.path.join(sys.argv[1], "all.pkl")
    # split_dataset(sys.argv[1], train_path, val_path, all_path)
    # plot_file_sizes(sys.argv[1])

    input_size = (200, 200)
    transform = Compose([
        ResizeCV(input_size),
        RandomDA(),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406],
        #                      [0.229, 0.224, 0.225])
    ])

    val_transform = Compose([
        ResizeCV(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_path = os.path.join("/home/etienneperot/workspace/datasets/snakes/train/val_and_google.pkl")
    dataset = SnakeDataset(train_path, transform = transform)
    dataloader = DataLoader(dataset, batch_size=4,
                            shuffle=True, num_workers=2,
                            pin_memory=True)
    for x, y, names in dataloader:
        assert(len(y) == len(names))
        z = unmake_grid((x*255).byte().cpu().numpy())

        # paranoia
        # idx = random.randint(0, len(names)-1)
        # img = cv2.imread(names[idx])
        # cv2.imshow('check'+str(idx), img)

        cv2.imshow("img", z)
        cv2.waitKey(10)