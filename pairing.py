'''
Filipe Chagas
10 - Feb - 2022
'''

import pandas as pd
import json
import os
import numpy as np
from tqdm import tqdm
from typing import *

RANDOM_STATE = 1

identity_df = pd.read_csv(os.path.join('celeba', 'identity_CelebA.txt'), sep=' ', header=None)

FILE_COLUMN = 0
PERSON_COLUMN = 1

def get_pairs(ids: Iterable) -> Tuple[pd.DataFrame, pd.DataFrame]:
    def get_genuine_pairs() -> pd.DataFrame:
        print('Generating genuine pairs')
        paired_rows = pd.DataFrame(columns=['file_a', 'person_a', 'file_b', 'person_b'])
        for id in tqdm(ids):
            my_rows = identity_df[identity_df[PERSON_COLUMN] == id]
            my_rows_a = pd.DataFrame({'file_a': my_rows[FILE_COLUMN], 'person_a': my_rows[PERSON_COLUMN]})
            my_rows_b = pd.DataFrame({'file_b': my_rows[FILE_COLUMN], 'person_b': my_rows[PERSON_COLUMN]})
            my_paired_rows = my_rows_a.merge(my_rows_b, how='cross')
            my_paired_rows = my_paired_rows[my_paired_rows['file_a'] != my_paired_rows['file_b']]
            if my_paired_rows.shape[0] > my_rows.shape[0]:
                my_paired_rows = my_paired_rows.sample(n=my_rows.shape[0], random_state=RANDOM_STATE) #ensures that the generated dataset has no more rows than the original dataset
            paired_rows = paired_rows.append(my_paired_rows, ignore_index=True)
        return paired_rows

    def get_impostor_pairs() -> pd.DataFrame:
        print('Generating imporstor pairs')
        paired_rows = pd.DataFrame(columns=['file_a', 'person_a', 'file_b', 'person_b'])
        for id in tqdm(ids):
            my_rows = identity_df[identity_df[PERSON_COLUMN] == id]
            other_rows = identity_df[identity_df[PERSON_COLUMN] != id].sample(n=my_rows.shape[0], random_state=RANDOM_STATE) #ensures that the generated dataset has no more rows than the original dataset
            my_rows_a = pd.DataFrame({'file_a': my_rows[FILE_COLUMN], 'person_a': my_rows[PERSON_COLUMN]})
            my_rows_b = pd.DataFrame({'file_b': other_rows[FILE_COLUMN], 'person_b': other_rows[PERSON_COLUMN]})
            my_paired_rows = my_rows_a.merge(my_rows_b, how='cross')
            if my_paired_rows.shape[0] > my_rows.shape[0]:
                my_paired_rows = my_paired_rows.sample(n=my_rows.shape[0], random_state=RANDOM_STATE) #ensures that the generated dataset has no more rows than the original dataset
            paired_rows = paired_rows.append(my_paired_rows, ignore_index=True)
        return paired_rows

    return get_genuine_pairs(), get_impostor_pairs()

if __name__ == '__main__':
    with open(os.path.join('celeba_partitions', 'partitions.json'), 'r') as f:
        partitions = json.load(f)

    if not os.path.exists('celeba_pairs'):
        os.mkdir('celeba_pairs')

    #Eval pairs
    print('EVAL')
    eval_genuine_pairs, eval_impostor_pairs = get_pairs(partitions['eval'])
    eval_genuine_pairs.to_csv(os.path.join('celeba_pairs', 'eval_genuine_pairs.csv'))
    eval_impostor_pairs.to_csv(os.path.join('celeba_pairs', 'eval_impostor_pairs.csv'))

    #Training pairs
    print('TRAINING')
    training_genuine_pairs, training_impostor_pairs = get_pairs(partitions['training'])
    training_genuine_pairs.to_csv(os.path.join('celeba_pairs', 'training_genuine_pairs.csv'))
    training_impostor_pairs.to_csv(os.path.join('celeba_pairs', 'training_impostor_pairs.csv'))