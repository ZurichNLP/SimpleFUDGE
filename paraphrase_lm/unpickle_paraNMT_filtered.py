#!/usr/bin/env python
# coding=utf-8

"""
Converts pickle dataset files from paraNMT_filtered
(https://drive.google.com/drive/folders/1Y54r47VSXP0Bo1h2cHhTMOLCH-YsP112;
Krishna et al. 2020: Reformulating Unsupervised Style Transfer as Paraphrase Generation) to csv
to simplify fine-tuning huggingface models.

Contents of pickles is a list of instance tuples.
Tuple fields correspond to the following:

temp1, temp2, equality, sent1, sent2, f1_scores, kt_scores, ed_scores, langids

(assumption based on
https://github.com/martiansideofthemoon/style-transfer-paraphrase/blob/master/datasets/parse_paranmt_postprocess.py#:~:text=all_temp1%2C%20all_temp2%2C%20all_equality%2C%20all_sent1%2C%20all_sent2%2C%20all_f1_scores%2C%20all_kt_scores%2C%20all_ed_scores%2C%20all_langids)
"""

from pathlib import Path
import pickle
import pandas as pd

pickle_path = Path('/srv/scratch6/kew/paraphrase/paraNMT/paranmt_filtered/')

for split in ['dev', 'train']:
    in_pkl_file = str(pickle_path / f'{split}.pickle')
    out_csv_file = str(pickle_path / f'{split}.csv')
    with (open(in_pkl_file, 'rb')) as f:
        data = pickle.load(f)
            
    # save to csv
    df = pd.DataFrame(data, columns=[
        'temp1',
        'temp2',
        'equality',
        'sent1',
        'sent2',
        'f1_scores',
        'kt_scores',
        'ed_scores',
        'langids'
        ])
    
    df.to_csv(out_csv_file, index=False)