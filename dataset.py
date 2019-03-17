from __future__ import print_function
import os
import sys
import glob
import random
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import data_augmentation as cv_da
import cv2

# some files are 0bytes...
def can_load_it(img_name):
    image = cv2.imread(img_name)
    return image is not None

def subselect(files):
    good = []
    for file in files:
        if can_load_it(file):
            good.append(file)
    return good

def split_dataset(directory, train_out, val_out, ratio=0.7):
    """
    will create 2 csv (train/ val)
    :param path:
    :return:
    """
    train_dic, val_dic = {}, {}
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
        val_dic[class_name] = val_dic

    pickle.dump(train_dic, open(train_out, "wb"))
    pickle.dump(val_dic, open(val_out, "wb"))


class SnakeDataset(Dataset):
    """Snake Facebook."""

    def __init__(self, pkl_file, imsize=(300, 300), transform=None):
        self.dic = pickle.load(open(pkl_file,'r'))
        self.get_list(self.dic)
        self.transform = transform
        self.imsize = imsize #resize to canonical size for now

    def get_list(self, dic):
        samples = []
        skeys = sorted(dic.keys(), key=lambda x:x.split('class-')[1])
        for c, key in enumerate(skeys):
            for file in dic[key]:
                samples.append((c,file))

        idx = range(len(samples))
        random.shuffle(idx)
        self.samples = [samples[i] for i in idx]
        print('SnakeDataset: ', len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        label, img_name = self.samples[idx]
        try:
            image = cv2.imread(img_name)
        except Exception as e:
            print(e)
        if image is None:
            print(img_name, ' is bad')
        image = cv2.resize(image, self.imsize, 0, 0, interpolation=cv2.INTER_AREA)
        h = cv_da.get_random_homography(height=self.imsize[1], width=self.imsize[0])
        image = cv_da.cv2_apply_transform_image(image, h)
        image = torch.from_numpy(image).permute([2,0,1])

        return (image, label)



if __name__ == '__main__':


    train_path = os.path.join(sys.argv[1], "train.pkl")
    val_path = os.path.join(sys.argv[1], "val.pkl")
    split_dataset(sys.argv[1], train_path, val_path)


    dataset = SnakeDataset(train_path)

    train_path = os.path.join("/home/etienneperot/workspace/datasets/snakes/train/train.pkl")
    dataset = SnakeDataset(train_path)
    dataloader = DataLoader(dataset, batch_size=8,
                            shuffle=True, num_workers=2,
                            pin_memory=True)

    for x, y in dataloader:

        z = cv_da.unmake_grid(x.byte().cpu().numpy())

        cv2.imshow("img", z)
        cv2.waitKey(5)