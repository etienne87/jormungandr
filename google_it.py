"""
Google for more images!
"""
from __future__ import print_function
import os
import glob
import pickle

from google_images_download import google_images_download
import utils


def create_googled_dataset(species_to_idx):
    response = google_images_download.googleimagesdownload()
    for key, value in species_to_idx.iteritems():
        arguments = {"keywords":key,"limit":100,"print_urls":True, "output_directory": output_directory}   #creating list of arguments
        paths = response.download(arguments)   #passing the arguments to the function

def get_pickle(directory, species_to_idx, out_filename):
    subdirs = sorted([x[0] for x in os.walk(directory) if x[0] != directory])
    dic = {}
    for subdir in subdirs:
        specie = os.path.basename(subdir)
        idx = species_to_idx[specie]
        idx = 'class-'+str(idx)
        files = glob.glob(subdir + '/*.jpg')
        out = []
        for item in files:
            if utils.can_load_it(item):
                out.append(item)
        dic[idx] = out
    pickle.dump(dic, open(out_filename, "wb"))


if __name__ == '__main__':
    species_to_idx = utils.get_species('data/class_id_mapping.csv')
    output_directory = '/home/etienneperot/workspace/datasets/snakes/google/'

    # create_googled_dataset(species_to_idx)
    get_pickle(output_directory, species_to_idx, output_directory + 'google.pkl')


    output_directory = '/home/etienneperot/workspace/datasets/snakes/train/'
    pkl1 = output_directory + 'google.pkl'
    pkl2 = output_directory + 'train.pkl'
    pkl3 = output_directory + 'val.pkl'
    pkl_merge_1 = output_directory + 'train_and_google.pkl'
    pkl_merge_2 = output_directory + 'val_and_google.pkl'

    utils.merge_pickle(pkl1, pkl2, pkl_merge_1)
    utils.merge_pickle(pkl1, pkl3, pkl_merge_2)

