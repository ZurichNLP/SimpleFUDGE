#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: Tannon Kew

"""
Example Call:

    python extract_sentences_from_wiki_corpus.py /srv/scratch6/kew/ats/data/en/simplewiki/simplewiki.txt /srv/scratch6/kew/ats/data/en/simplewiki/simplewiki_sents.txt

"""

import sys
from itertools import islice

import spacy
    
# setup spacy sentencizer
nlp = spacy.load("en_core_web_sm", disable=["tagger", "ner", "lemmatizer"]) # just the parser
nlp.add_pipe('sentencizer')

def split_sentences_for_batched_lines(file, n=256, min_length=5):
    with open(file, 'r', encoding='utf-8') as f:
        for n_lines in iter(lambda: tuple(islice(f, n)), ()):            
            for doc in nlp.pipe(n_lines, batch_size=n):
                for sent in doc.sents:
                    if len(sent) > min_length:
                        yield sent.text.strip()
                    
    
if __name__ == '__main__':
    
    infile = sys.argv[1]
    outfile = sys.argv[2]
    
    c = 0
    with open(outfile, 'w', encoding='utf8') as outf:
        for sent in split_sentences_for_batched_lines(infile):
            outf.write(sent + '\n')
            c += 1
            if c % 10000 == 0:
                print(f'progress: {c} sentences ...')
        # print(sent)
    print(f'Finished writing {c} sentences to {outfile}')
    
        