from __future__ import print_function
import os
import sys
import glob
import random
import pickle
from torch.utils.data import Dataset, DataLoader
import data_augmentation as da
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
        val_dic[class_name] = val_files

    pickle.dump(train_dic, open(train_out, "wb"))
    pickle.dump(val_dic, open(val_out, "wb"))


class ResizeCV(object):
    def __init__(self, imsize):
        self.imsize = imsize

    def __call__(self, sample):
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
        self.get_list(self.dic)
        self.transform = transform

    def get_list(self, dic):
        samples = []
        skeys = sorted(dic.keys(), key=lambda x:x.split('class-')[1])
        for c, key in enumerate(skeys):
            for file in dic[key]:
                print(file)
                samples.append((c,file))

        idx = range(len(samples))
        random.shuffle(idx)
        self.samples = [samples[i] for i in idx]

    def __len__(self):
        return len(self.samples) // 1000

    def __getitem__(self, idx):
        label, img_name = self.samples[idx]
        image = cv2.imread(img_name)
        if image is None:
            print('Could not read: ', img_name)
            assert 0
        if self.transform is not None:
            image = self.transform(image)
        return (image, label)



if __name__ == '__main__':
    from torchvision import transforms

    train_path = os.path.join(sys.argv[1], "train.pkl")
    val_path = os.path.join(sys.argv[1], "val.pkl")
    split_dataset(sys.argv[1], train_path, val_path)

    input_size = (300, 300)
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

    train_path = os.path.join("/home/etienneperot/workspace/datasets/snakes/train/val.pkl")
    dataset = SnakeDataset(train_path, transform = transform)
    dataloader = DataLoader(dataset, batch_size=8,
                            shuffle=True, num_workers=2,
                            pin_memory=True)
    for x, y in dataloader:
        z = da.unmake_grid((x*255).byte().cpu().numpy())

        cv2.imshow("img", z)
        cv2.waitKey(500)