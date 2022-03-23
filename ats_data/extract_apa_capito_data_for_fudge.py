#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example call:

    # sentence level
    python extract_apa_capito_data_for_fudge.py --in_dir /srv/scratch6/kew/ats/data/de/apa_capito/ --splits /srv/scratch6/kew/ats/data/de/aligned --format sentence

    # doc / paragraph level
    python extract_apa_capito_data_for_fudge.py --in_dir /srv/scratch6/kew/ats/data/de/apa_capito/ --splits /srv/scratch6/kew/ats/data/de/aligned --format paragraph

NOTE: This does not guarantee that there is absolutely no overlap between splits due to slight differences in tokenization.

e.g. Pre-tokenized aligned sentences are split at colons:

'Denn mit dem persönlichen Budget entscheiden Sie selbst , welche Unterstützungen Sie beim Wohnen und in Ihrer Freizeit brauchen :', 'Sie entscheiden , wobei Sie Unterstützung brauchen .'

vs. 

'Denn mit dem persönlichen Budget entscheiden Sie selbst, welche Unterstützungen Sie beim Wohnen und in Ihrer Freizeit brauchen: Sie entscheiden, wobei Sie Unterstützung brauchen.'

However, the effect is relatively mild (~3-7 overlap texts in each split). Can check this with `overlap_sanity_check.py`

"""

import argparse
from pathlib import Path
import pandas as pd
import nltk
from tqdm import tqdm
import random
import sklearn
import string
import re

def set_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in_dir', required=True, type=Path, default=None, help='path to directory containing apa_v6 and capito_v6 corpora')
    ap.add_argument('--out_dir', required=False, type=Path, default=None, help='path to output directory')
    ap.add_argument('--splits', required=False, type=Path, default=None, help='path to directory containing split tsv files. Used to remove overlapping sentences')
    ap.add_argument('--format', required=True, type=str, choices=['sentence', 'paragraph'], default='sentence', help='whether or not to write one sentence per line or retain some sequences of sentences (experimental)')
    return ap.parse_args()

def read_articles(filepath):
    """ keeps paragraph structure """
    para_sents = []
    with open(filepath, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip()
            if line:
                para_sents += [nltk.sent_tokenize(line)]           
    return para_sents

def read_tsv(filepath):
    src_sents, tgt_sents = [], []
    with open(filepath, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip().split('\t')
            src_sents.append(line[0])
            tgt_sents.append(line[1])
    return src_sents, tgt_sents

def strip_whitespaces(string):
    return re.sub('\s+', '', string)

def is_valid_sentence(sentence):

    if len(sentence.split()) < 3:
        return False
    
    bad_chars = sum(1 for c in sentence if c in string.digits or c in string.punctuation or c in string.whitespace)
    if bad_chars / len(sentence) > 0.4:
        return False

    return True

def get_v6_doc_aligned(data_dir, level):
    """
    apa_v6 and capito_v6 share the same directory structure
    """
    data_dir = data_dir / f'7_document-aligned/dms_v5/'

    orig_texts = []
    simp_texts = []

    for sub_dir in tqdm(sorted(data_dir.iterdir())):
        orig_filepath = sub_dir / f'{level}-OR' / f'all{level}-OR.de'
        simp_filepath = sub_dir / f'{level}-OR' / f'all{level}-OR.simpde'

        if orig_filepath.exists() and simp_filepath.exists():
            orig_texts += read_articles(orig_filepath)
            simp_texts += read_articles(simp_filepath)

    return orig_texts, simp_texts

def remove_test_sentences_complete_docs(test_file, orig_texts, simp_texts):
    """
    removes complete src-tgt documents texts if one sentence from either text appears in test set
    """
    src_sents, tgt_sents = read_tsv(test_file)
    src_sents_ = [strip_whitespaces(sent).lower() for sent in src_sents]
    tgt_sents_ = [strip_whitespaces(sent).lower() for sent in tgt_sents]

    orig_texts_ = []
    for text in orig_texts:
        text = [strip_whitespaces(sent).lower() for sent in text]
        orig_texts_.append(text)

    simp_texts_ = []
    for text in simp_texts:
        text = [strip_whitespaces(sent).lower() for sent in text]
        simp_texts_.append(text)

    items_to_remove = []
    for i, text in tqdm(enumerate(orig_texts_), total=len(orig_texts_)):
        for sent in text:
            if sent in src_sents_:
                items_to_remove.append(i)
    for i, text in tqdm(enumerate(simp_texts_), total=len(simp_texts_)):
        for sent in text:
            if sent in tgt_sents_:
                items_to_remove.append(i)
    
    orig_texts = [orig_texts[i] for i in range(len(orig_texts)) if i not in items_to_remove]
    simp_texts = [simp_texts[i] for i in range(len(simp_texts)) if i not in items_to_remove]

    return orig_texts, simp_texts

def split_docs_at_problematic_sentences(texts, prob_texts_id):
    new_texts = []
    for txt_idx in range(len(prob_texts_id)):
        if len(prob_texts_id[txt_idx]):
            prob_text = texts[txt_idx] # get the problematic text that we will truncate
            # breakpoint()
            # for s in 
            if 'Sie wohnen zu Hause.' in prob_text:
                breakpoint()

            cur_idx = 0 # init first index as zero
            for sent_idx in prob_texts_id[txt_idx]:
                trunc_prob_text = prob_text[cur_idx:sent_idx] # truncate text at sent_idx
                cur_idx = sent_idx+1 # update the current index for next truncation starting point (if applicable)
                if len(trunc_prob_text): # only append truncated docs if they are non-empty, i.e. where cur_idx and sent_idx are equal
                    new_texts.append(trunc_prob_text)
        else:
            new_texts.append(texts[txt_idx])
    return new_texts

def id_problematic_sents(texts, sents):
    """
    texts: witespace removed and lowercased sentence-tokenized texts/docs
    sents: witespace removed and lowercased sentences in test split
    """
    prob_texts = []
    for text in tqdm(texts, total=len(texts)):
        sents_to_remove = []
        for j, sent in enumerate(text):
            if sent in sents:
                sents_to_remove.append(j)
        prob_texts.append(sents_to_remove)
    assert len(prob_texts) == len(texts)
    return prob_texts

def remove_test_sentences(test_file, orig_texts, simp_texts):
    """
    removes sentences from src-tgt document texts that appear in test set.
    given a doc [s1, s2, s3, s4, s5], if s4 appears in test set, we get
    doc1 [s1, s2, s3] and doc2 [s5]
    """
    src_sents, tgt_sents = read_tsv(test_file)
    src_sents_ = [strip_whitespaces(sent).lower() for sent in src_sents] # witespace removed and lowercased
    tgt_sents_ = [strip_whitespaces(sent).lower() for sent in tgt_sents] # witespace removed and lowercased

    orig_texts_ = [] # witespace removed and lowercased
    for text in orig_texts:
        orig_texts_.append([strip_whitespaces(sent).lower() for sent in text])
        
    simp_texts_ = [] # witespace removed and lowercased
    for text in simp_texts:
        simp_texts_.append([strip_whitespaces(sent).lower() for sent in text])
    
    prob_src_texts = id_problematic_sents(orig_texts_, src_sents_)
    prob_tgt_texts = id_problematic_sents(simp_texts_, tgt_sents_)
    
    orig_texts = split_docs_at_problematic_sentences(orig_texts, prob_src_texts)    
    simp_texts = split_docs_at_problematic_sentences(simp_texts, prob_tgt_texts)
    
    return orig_texts, simp_texts


def clean_docs(orig_texts, simp_texts):
    """
    removes
        - short documents
        - docs where src and tgt are copies
    """
    items_to_remove = []
    for i, (orig_text, simp_text) in tqdm(enumerate(zip(orig_texts, simp_texts))):
        if len(orig_text) < 3 or len(simp_text) < 3: # remove docs containing only 1 or sentences
            items_to_remove.append(i)
        if orig_text == simp_text: # remove copies
            items_to_remove.append(i)

    orig_texts = [orig_texts[i] for i in range(len(orig_texts)) if i not in items_to_remove]
    simp_texts = [simp_texts[i] for i in range(len(simp_texts)) if i not in items_to_remove]

    return orig_texts, simp_texts

def shuf_and_split(data):
    random.Random(42).shuffle(data)
    train_data, test_data = sklearn.model_selection.train_test_split(data, train_size=0.8, test_size=0.2, shuffle=False)
    test_data, val_data = sklearn.model_selection.train_test_split(test_data, train_size=0.5, test_size=0.5, shuffle=False)
    return train_data, val_data, test_data

if __name__ == '__main__':

    args = set_args()

    for level in ['A1', 'A2', 'B1']:
        orig_texts, simp_texts = [], []
        src, tgt = get_v6_doc_aligned(args.in_dir / 'capito_v6', level)
        print(f'Aligned docs for OR-{level} from CAPITO: {len(src)}-{len(tgt)}')
        orig_texts += src
        simp_texts += tgt
        if level != 'A1':
            src, tgt = get_v6_doc_aligned(args.in_dir / 'apa_v6', level)
            print(f'Aligned docs for OR-{level} from APA: {len(src)}-{len(tgt)}')
            orig_texts += src
            simp_texts += tgt
        
        print(f'All aligned docs for OR-{level} yields {len(orig_texts)}-{len(simp_texts)}')
        
        # simple doc removal
        orig_texts, simp_texts = clean_docs(orig_texts, simp_texts)
        print(f'Removing short/duplicate docs yields {len(orig_texts)}-{len(simp_texts)}')

        # if splits are provided, remove test/dev sentences from docs 
        if args.splits:
            for split in ['test', 'dev']:
                split_file = args.splits / f'apa_capito_{level.lower()}_{split}.tsv'
                if split_file.exists():
                    orig_texts, simp_texts = remove_test_sentences(split_file, orig_texts, simp_texts)
                    print(f'Removing {split} set sentences and splitting docs yields {len(orig_texts)}-{len(simp_texts)}')
        
        
        if args.out_dir is not None:
            out_dir = args.out_dir
        else:
            out_dir = args.in_dir / f'article_{args.format}s'
        
        out_dir.mkdir(parents=True, exist_ok=True)

        orig_train, orig_dev, orig_test = shuf_and_split(orig_texts)
        simp_train, simp_dev, simp_test = shuf_and_split(orig_texts)

        for split, data in {'train': orig_train, 'dev': orig_dev, 'test': orig_test}.items():
            src_outfile = out_dir / f'{split}_or-{level}.de'
            with open(src_outfile, 'w', encoding='utf8') as src_outf:
                c = 0
                for text in data:
                    if args.format == 'sentence':
                        for sent in text:
                            if is_valid_sentence(sent):
                                src_outf.write(f'{sent}\n')
                                c += 1
                    else:        
                        src_outf.write(f'{" ".join(text)}\n')
                        c += 1
                print(f'Wrote {c} {args.format}s to {src_outfile} ...')
                
        for split, data in {'train': simp_train, 'dev': simp_dev, 'test': simp_test}.items():
            tgt_outfile = out_dir / f'{split}_or-{level}.simpde'
            with open(tgt_outfile, 'w', encoding='utf8') as tgt_outf:
                c = 0
                for text in data:
                    if args.format == 'sentence':
                        for sent in text:
                            if is_valid_sentence(sent):
                                tgt_outf.write(f'{sent}\n')
                                c += 1
                    else:        
                        tgt_outf.write(f'{" ".join(text)}\n')
                        c += 1
                print(f'Wrote {c} {args.format}s to {tgt_outfile} ...')