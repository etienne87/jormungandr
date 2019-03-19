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
        specie = os.path.dirname(subdir)
        idx = species_to_idx[specie]
        dic[idx] = glob.glob(subdir + '/*.jpg')
    pickle.dump(dic, open(out_filename, "wb"))


if __name__ == '__main__':
    species_to_idx = utils.get_species('data/class_id_mapping.csv')
    output_directory = '/home/etienneperot/workspace/datasets/snakes/google/'

    # create_googled_dataset(species_to_idx)
    get_pickle(output_directory, species_to_idx, output_directory + 'google.pkl')

