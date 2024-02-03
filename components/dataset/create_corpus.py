import os
from tqdm import tqdm
from glob import glob

from utils.preprocessing import preprocessing
from utils.io import find_folders, load_file


def create_corpus(folder_path):
    corpus = []

    sub_folders = find_folders(folder_path)

    for sub_folder in tqdm(sub_folders):
        print(f'Running on: {sub_folder}')

        files = glob(os.path.join(sub_folder, '*.txt'))

        for file in files:
            data = load_file(file)
            corpus.extend(data)

    corpus = [x for x in corpus if len(x) >= 1]
    corpus = list(map(preprocessing, corpus))
    corpus = list(set(corpus))
    return corpus
