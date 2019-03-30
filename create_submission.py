from __future__ import print_function
import os
import glob
import csv
import pickle
import dataset
import models
from torchvision import transforms
import torch
import torch.nn.functional as F
import utils
from dataset import subselect

def folder_to_pkl(dir, test_out):
    files = glob.glob(dir + '/*.jpg')
    files = subselect(files)
    dic = {}
    dic['class-0'] = files
    pickle.dump(dic, open(test_out, "wb"))


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

def run_dataset(file_path, checkpoint, idx_to_species, cuda):
    # Load Data
    input_size = (600, 600)
    mean = [0.485, 0.456, 0.406][::-1]
    std = [0.229, 0.224, 0.225][::-1]

    data_transform = dataset.Compose([
        dataset.ResizeCV(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    batch_size = 16
    ds = dataset.SnakeDataset(file_path, data_transform)
    dataloader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)

    # Load model & checkpoint
    model = models.resnet(num_classes=45, pretrained=True, resnet_model='resnet18', add_stn=False)
    model = utils.load_model(checkpoint, model)

    if cuda:
        model = model.cuda()

    model = model.eval()

    submit = []
    keys = ['filename']
    for i in range(len(idx_to_species)):
        keys += [idx_to_species[i]]
    print(keys)
    submit.append(keys)
    for batch_idx, (x, _, names) in enumerate(dataloader):
        print(batch_idx, '/', len(ds)/batch_size)
        if cuda:
            x = x.cuda()
        outputs = model(x)
        scores = F.softmax(outputs,dim=1).cpu().data.numpy()
        for i, name in enumerate(names):
            row = [os.path.basename(name)]
            for j in range(scores[i].shape[0]):
                row += [str(scores[i, j])]
            submit.append(row)

    submission = file_path.replace('.pkl', '.csv')
    with open(submission, 'wb') as  csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for row in submit:
            writer.writerow(row)



if __name__ == '__main__':
    directory = '/home/etienneperot/workspace/datasets/snakes/train/'
    mapping_filename = 'data/class_id_mapping.csv'
    idx_to_class_name = get_map(directory, mapping_filename)

    dir_test = '/home/etienneperot/workspace/datasets/snakes/round1_test/'
    test_out = "/home/etienneperot/workspace/datasets/snakes/round1_test/test.pkl"
    #folder_to_pkl(dir_test + 'round1/', test_out)

    checkpoint = 'checkpoints/dummy_resnet18.pth'
    run_dataset(test_out, checkpoint, idx_to_class_name, cuda=True)

