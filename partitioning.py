import helper
import pandas as pd
import numpy as np
import json
import os

DENOMINATOR = 4  #|training|=(total*(denominator-1))//denominator  |eval|=total//denominator

identity_df = pd.read_csv(os.path.join('celeba','identity_CelebA.txt'), sep=' ', header=None)

FILE_COLUMN = 0
PERSON_COLUMN = 1

def get_persons_ids():
    return np.unique(identity_df[PERSON_COLUMN])

if __name__ == '__main__':
    ids = get_persons_ids()
    np.random.shuffle(ids)
    
    fraction = ids.shape[0]//DENOMINATOR
    eval_ids = ids[:fraction]
    training_ids = ids[fraction:]

    if not os.path.exists('celeba_partitions'):
        os.mkdir('celeba_partitions')

    with open(os.path.join('celeba_partitions', 'partitions.json'), 'w', encoding='utf-8') as file:
        json.dump({'eval': [int(x) for x in eval_ids], 'training': [int(x) for x in training_ids]}, file,  ensure_ascii=False, indent=4)
