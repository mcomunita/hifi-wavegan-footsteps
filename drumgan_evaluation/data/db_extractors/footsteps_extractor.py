import os
import argparse
# import hashlib
import sys
# import pickle
# import requests

# from drumgan_evaluation.utils.utils import get_date, mkdir_in_path, read_json, list_files_abs_path, get_filename, save_json
# from utils.utils import get_date, mkdir_in_path, read_json, list_files_abs_path, get_filename, save_json

# from random import shuffle
# from tqdm import trange, tqdm

# from .base_db import get_base_db, get_hash_dict

# import ipdb
import numpy as np


def extract(path: str, criteria: dict={}):
    if not os.path.exists(path):
        print('Footsteps folder not found')
        print(path)
        sys.exit(1)
    data = []
    labels_names = []

    for folder in sorted(next(os.walk(path))[1]):
        for wavfile in os.listdir(path+folder):
            item_path = '%s%s/%s' % (path, folder, wavfile)
            data.append(item_path)
            labels_names.append(folder)
    labels = list(np.unique(labels_names, return_inverse=True)[1])

    return data, labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Footsteps database extractor')
    parser.add_argument('footsteps_path', type=str,
                         help='Path to the footsteps root folder')
    
    args = parser.parse_args()
    
    extract(path=args.footsteps_path,
            criteria={})
