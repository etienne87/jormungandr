# pylint: disable-all
from __future__ import print_function
import os
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

def random_rotate(rotation_range):
    degree = random.uniform(-rotation_range, rotation_range)
    theta = math.pi / 180 * degree
    rotation_matrix = np.array([[math.cos(theta), -math.sin(theta), 0],
                                [math.sin(theta), math.cos(theta), 0],
                                [0, 0, 1]])
    return rotation_matrix


def random_translate(height_range, width_range):
    tx = random.uniform(-height_range, height_range)
    ty = random.uniform(-width_range, width_range)
    translation_matrix = np.array([[1, 0, tx],
                                   [0, 1, ty],
                                   [0, 0, 1]])
    return translation_matrix


def random_shear(shear_range):
    shear = random.uniform(-shear_range, shear_range)
    shear_matrix = np.array([[1, -math.sin(shear), 0],
                             [0, math.cos(shear), 0],
                             [0, 0, 1]])
    return shear_matrix


def random_zoom(zoom_range):
    z = random.uniform(zoom_range[0], zoom_range[1])
    zoom_matrix = np.array([[z, 0, 0],
                            [0, z, 0],
                            [0, 0, 1]])
    return zoom_matrix


def random_horizontal_flip():
    if np.random.randint(2):
        mat = np.array([[-1, 0, 1],
                        [0, 1, 0],
                        [0, 0, 1]], dtype=np.float32)
    else:
        mat = np.eye(3)
    return mat


def affine_compose(tforms):
    tform_matrix = tforms[0]
    for tform in tforms:
        tform_matrix = np.dot(tform_matrix, tform)
    return tform_matrix


def get_affine_matrix():
    tforms = []
    tforms.append(random_rotate(2))
    tforms.append(random_translate(-0.1, 0.1))
    tforms.append(random_horizontal_flip())
    tforms.append(random_zoom((0.7, 1.3)))
    return affine_compose(tforms)


def get_random_homography(height=1, width=1, perspective_range=1e-6):
    mat = np.eye(3) #+ np.random.randn(3, 3) * perspective_range
    mat = np.dot(mat, get_affine_matrix())
    mat[0, 2] *= width
    mat[1, 2] *= height
    return mat


def cv2_apply_transform_image(image, transform, borderValue=(127, 127, 127)):
    """
    Applies homography to an image
    :param batch: H, W, C
    :param transforms:
    :param borderValue:
    :return:
    """
    h, w = image.shape[0], image.shape[1]
    image = cv2.warpPerspective(image, transform, (w, h), flags=cv2.INTER_LINEAR, borderValue=borderValue)
    return image

def cv2_apply_transform_batch(batch, transforms, borderValue=(70, 70, 70)):
    """
    Applies N different homographies
    :param batch: N H W C
    :param tranform:
    :param borderValue:
    :return:
    """
    for i in range(batch.shape[0]):
        batch[i] = cv2_apply_transform_image(batch[i], transforms[i], borderValue)
    return batch


class Affine(nn.Module):
    """
    Data augmentation in Pytorch
    Applies Warping on the images by selecting a random transformation.

    """

    def __init__(self, batchsize=32, height=240, width=304, use_homography=True):
        super(Affine, self).__init__()
        self.batchsize, self.height, self.width = batchsize, height, width
        self.use_homography = use_homography
        self.reset_params()

    def reset_params(self):
        thetas = []
        invthetas = []

        for i in range(self.batchsize):
            if self.use_homography:
                theta = get_random_homography()
            else:
                theta = get_affine_matrix()

            theta2 = np.linalg.inv(theta)
            theta = torch.from_numpy(theta).float()
            theta2 = torch.from_numpy(theta2).float()
            thetas.append(theta.unsqueeze(0))
            invthetas.append(theta2.unsqueeze(0))

        invthetas = torch.cat(invthetas)
        thetas = torch.cat(thetas)

        grid_h, grid_w = torch.meshgrid([torch.linspace(-1., 1., self.height),
                                         torch.linspace(-1., 1., self.width)])
        grid = torch.cat((grid_w[None, :, :, None],
                          grid_h[None, :, :, None]), 3)

        grid = grid.repeat(self.batchsize, 1, 1, 1)
        for i in range(self.batchsize):
            grid_ncd = grid[i].view(-1, 2)
            warped_grid = torch.mm(grid_ncd, invthetas[i, :2, :]) + invthetas[i, 2]
            if self.use_homography:
                warped_grid = warped_grid / warped_grid[:, 2:3]
            warped_grid = warped_grid[:, :2]
            grid[i] = warped_grid.view(1, self.height, self.width, 2)

        if hasattr(self, "grid"):
            self.grid[...] = grid
        else:
            self.register_buffer("grid", grid)

        if hasattr(self, "theta"):
            self.theta[...] = thetas
        else:
            self.register_buffer("theta", thetas)

    def warp_images(self, x):
        grid = self.grid[:x.size(0)]
        y = F.grid_sample(x, grid)
        return y

    def forward(self, x):
        y = self.warp_images(x)
        return y


if __name__ == '__main__':
    from dataset import SnakeDataset
    from torch.utils.data import Dataset, DataLoader
    from utils import unmake_grid


    train_path = os.path.join("/home/etienneperot/workspace/datasets/snakes/train/train.pkl")
    dataset = SnakeDataset(train_path)
    dataloader = DataLoader(dataset, batch_size=8,
                            shuffle=True, num_workers=2,
                            pin_memory=True)

    data_aug = Affine(batchsize=8, height=300, width=300)
    for x, y in dataloader:
        x2 = data_aug(x.float())

        z = unmake_grid(x2.byte().cpu().numpy())

        cv2.imshow("img", z)
        cv2.waitKey(5)
