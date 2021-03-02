import pandas as pd
from common import base_data_path
import os


def generate_vocab():
    filepath = os.path.join(base_data_path, "dataset_v1.csv")
    df = pd.read_csv(filepath, sep=",", header=0)
    vocab = {}

    multi_char = ['Br', 'Cl']

    for _i, row in df.iterrows():
        smile_repr = row[0]
        new_el = 0
        skip_next = False
        for i in range(len(smile_repr)):
            if skip_next:
                skip_next = False
                continue

            element = smile_repr[i]
            if (i < (len(smile_repr) - 1)) and (smile_repr[i].isalpha()):
                element_1 = element + smile_repr[i+1]
                if element_1 in multi_char:
                    element = element_1
                    skip_next = True
            if element not in vocab:
                new_el += 1
            vocab[element] = 1

    vocab = list(vocab.keys())
    vocab.sort()

    filepath = os.path.join(base_data_path, "vocab-2705.txt")
    f = open(filepath, "w")
    for element in vocab:
        f.write(element+'\n')
    f.close()
    pass


generate_vocab()
pass
