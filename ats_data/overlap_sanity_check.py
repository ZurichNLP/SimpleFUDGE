#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

tokenizer.tokenize('Durch die Auflösung Ihres Zustellprofils werden Sie nur für elektronische Zustellungen gemäß Zustellgesetz abgemeldet .')
['▁Durch', '▁die', '▁Auflösung', '▁Ihres', '▁Zu', 'stell', 'profil', 's', '▁werden', '▁Sie', '▁nur', '▁für', '▁elektronische', '▁Zu', 
'stellung', 'en', '▁gemäß', '▁Zu', 'stell', 'gesetz', '▁abge', 'meld', 'et', '▁', '.']

tokenizer.tokenize('Durch die Auflösung Ihres Zustellprofils werden Sie nur für elektronische Zustellungen gemäß Zustellgesetz abgemeldet.')
['▁Durch', '▁die', '▁Auflösung', '▁Ihres', '▁Zu', 'stell', 'profil', 's', '▁werden', '▁Sie', '▁nur', '▁für', '▁elektronische', '▁Zu', 
'stellung', 'en', '▁gemäß', '▁Zu', 'stell', 'gesetz', '▁abge', 'meld', 'et', '.']

"""

from pathlib import Path
import sys
import re
from transformers import MBartTokenizer
import nltk
from itertools import chain

from extract_apa_capito_data_for_fudge import read_articles, read_tsv

splits_path = Path('/srv/scratch6/kew/ats/data/de/aligned') #sys.argv[1] # /srv/scratch6/kew/ats/data/de/aligned/apa_capito_a1_dev.tsv
sents_path = Path('/srv/scratch6/kew/ats/data/de/apa_capito/article_sentences')
# sents_path = Path('/srv/scratch6/kew/ats/data/de/apa_capito/article_paragraphs')

levels = ['A1', 'A2', 'B1']

# tokenizer = MBartTokenizer.from_pretrained('/srv/scratch6/kew/ats/fudge/generators/mbart/longmbart_model_w512_20k')

def normalise_text(text):
    return re.sub('\s+', '', text.lower())


for level in levels:

    src_dev_test_sents = set()
    tgt_dev_test_sents = set()
    for split in ['dev', 'test']:
        src, tgt = read_tsv(splits_path / f'apa_capito_{level.lower()}_{split}.tsv')
        # breakpoint()
        src_dev_test_sents.update(list(map(normalise_text, src)))
        tgt_dev_test_sents.update(list(map(normalise_text, src)))

    src_train_sents = read_articles(sents_path / f'train_or-{level}.de')
    tgt_train_sents = read_articles(sents_path / f'train_or-{level}.simpde')
    
    src_train_sents = set(map(normalise_text, chain.from_iterable(src_train_sents)))
    tgt_train_sents = set(map(normalise_text, chain.from_iterable(tgt_train_sents)))
    
    # for i in dev_test_sents:
    #     if i in train_texts
    src_ol = src_dev_test_sents.intersection(src_train_sents)
    tgt_ol = tgt_dev_test_sents.intersection(tgt_train_sents)

    # breakpoint()
    if len(src_ol) > 0:
        print(f'[!] Found SRC overlap for {level}')
        print(src_ol)
        # breakpoint()
    else:
        print(f'No SRC overlap found for {level} :)')

    if len(tgt_ol) > 0:
        print(f'[!] Found TGT overlap for {level}')
        print(tgt_ol)
        
    else:
        print(f'No TGT overlap found for {level} :)')