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
    python data_prep/fasttext_classifier.py 3 0 1 2
"""

import sys
from pathlib import Path
import random
import argparse
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
            if line:
                texts.append(f'__label__{class_label}\t{line}')
    return texts

def write_to_tmp_outfile(data, filepath):
    with open(filepath, 'w', encoding='utf8') as f:
        for item in data:
            f.write(item+'\n')
    return

def train(file, epoch=25, lr=0.2):
    model = fasttext.train_supervised(file, epoch=epoch, lr=lr)
    return model

def print_results(N, p, r):
    print("N\t" + str(N))
    print("P@{}\t{:.3f}".format(1, p))
    print("R@{}\t{:.3f}".format(1, r))

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', required=True, type=str, help='data directory')
    parser.add_argument('-p', '--pos_class', required=True, type=int, help='positive class')
    parser.add_argument('-n', '--neg_classes', required=False, type=int, nargs='+', help='negative classes. If none given, all other classes are used as negative classes.')
    parser.add_argument('-s', '--seed', required=False, type=int, default=42, help='random seed')
    parser.add_argument('-c', '--clean_up', action='store_true', help='clean up tmp files after training and evaluation')
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    # pos_class = sys.argv[1]
    # neg_classes = sys.argv[2:]

    args = set_args()

    random.seed(args.seed)

    # indir = Path('resources/data/en/newsela_article_corpus_2016-01-29/article_sentences/en')
    tmpdir = Path(args.data_dir) / 'tmp'
    tmpdir.mkdir(parents=True, exist_ok=True)

    print('preparing data for fasttext...')
    
    if not args.neg_classes:
        files = Path(args.data_dir).glob('*_train.txt')
        args.neg_classes = [int(f.stem.split('_')[0]) for f in files if int(f.stem.split('_')[0]) != args.pos_class]
        print('Inferred negative classes: ', args.neg_classes)

    prep_data = {}
    for split in ['train', 'test', 'valid']:
        # pos class

        p_file = Path(args.data_dir) / f'{args.pos_class}_{split}.txt'
        pdata = preprocess_data(p_file, 1)

        # neg class(es)        
        n_files = [Path(args.data_dir) / f'{neg_class}_{split}.txt' for neg_class in args.neg_classes]

        ndata = []
        for n_file in n_files:
            if n_file.exists():
                ndata += preprocess_data(n_file, 0)
        random.shuffle(ndata)
        
        # balance out data
        if len(pdata) > len(ndata):
            pdata = pdata[:len(ndata)]
        else:
            ndata = ndata[:len(pdata)]

        data = pdata + ndata
        random.shuffle(data)

        prep_data[split] = str(Path(tmpdir) / f'{split}_{args.pos_class}_{"".join(map(str, args.neg_classes))}.txt')
        write_to_tmp_outfile(data, prep_data[split])

    # print(f'Simp. level {args.pos_class} vs. {" ".join(map(str, args.neg_classes))}')
    
    print(f'training fasttext model on {prep_data["train"]} ...')
    model = train(prep_data['train'])
    print(f'evaluating model on {prep_data["valid"]} ...')
    print_results(*model.test(prep_data['valid']))
    print(f'evaluating model on {prep_data["test"]} ...')
    print_results(*model.test(prep_data['test']))

    # clean up tmp files
    if args.clean_up:
        for file in tmpdir.iterdir():
            file.unlink()
