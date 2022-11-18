#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script for inspecting quick and dirty classifiers on
simplification level text classification.

Script expects :
    - sentences corresponding to a positive class (e.g.
    simplification level 4 according to Newsela)
    - sentences corresponding to one or more negative classes 

Given the input data, we prepare texts for fasttext and then
train and evaluate a binary fasttext classifier.

NOTE: model training and eval takes only a couple of
seconds, so we do not save the model files.

Example Call (positive class = 3, negative classes = 0, 1, 2):
    python ats_data/fasttext_classifier.py 3 0 1 2
"""

import sys
from pathlib import Path
import random
import string

import fasttext

random.seed(42)

from transformers import AutoTokenizer

tokenizer_name = 'facebook/bart-base'
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

def preprocess_data(file, class_label, tokenize=False):
    texts = []
    with open(file, 'r', encoding='utf8') as f:
        for line in f:
            if tokenize:
                line = ' '.join(tokenizer.encode(line.strip(), add_special_tokens=False))
            else:
                line = line.strip()
            texts.append(f'__label__{class_label}\t{line}')
    return texts

def write_to_tmp_outfile(data, filepath):
    with open(filepath, 'w', encoding='utf8') as f:
        for item in data:
            f.write(item+'\n')
    return

def train(file, epoch=25):
    model = fasttext.train_supervised(file, epoch=epoch, lr=1.0)
    # print(model.labels)
    return model

def print_results(N, p, r):
    print("N\t" + str(N))
    print("P@{}\t{:.3f}".format(1, p))
    print("R@{}\t{:.3f}".format(1, r))

if __name__ == '__main__':

    pos_class = sys.argv[1]
    neg_classes = sys.argv[2:]

    indir = Path('/srv/scratch6/kew/ats/data/en/newsela_article_corpus_2016-01-29/article_sentences/en')
    tmpdir = Path('/srv/scratch6/kew/ats/data/en/newsela_article_corpus_2016-01-29/article_sentences/fasttext/')
    tmpdir.mkdir(parents=True, exist_ok=True)

    print('preparing data for fasttext...')
    for split in ['train', 'test', 'valid']:
        # pos class
        p_file = indir / f'{split}_{pos_class}.txt'
        pdata = preprocess_data(p_file, 1)

        # neg class(es)        
        n_files = [str(indir / f'{split}_{neg_class}.txt') for neg_class in neg_classes]
        ndata = []
        for n_file in n_files:
            ndata += preprocess_data(n_file, 0)
        random.shuffle(ndata)
        
        # balance out data
        if len(pdata) > len(ndata):
            pdata = pdata[:len(ndata)]
        else:
            ndata = ndata[:len(pdata)]

        data = pdata + ndata
        random.shuffle(data)
        write_to_tmp_outfile(data, tmpdir / f'{split}_{pos_class}_{"".join(neg_classes)}.txt')

    print(f'Simp. level {pos_class} vs. {" ".join(neg_classes)}')
    model = train(str(tmpdir / f'train_{pos_class}_{"".join(neg_classes)}.txt'))
    print_results(*model.test(str(tmpdir / f'valid_{pos_class}_{"".join(neg_classes)}.txt')))

    # clean up tmp files
    for file in tmpdir.iterdir():
        file.unlink()
