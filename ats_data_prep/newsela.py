#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import random
import pandas as pd

SEED = 42
random.seed(SEED) # set seed for reproducibility
# tgt_lang = 'en'

def assign_newsela_splits(csv_file):
    """
    assigns test/train/valid split labels to articles in the
    Newsela corpus and splits them into langauge-specific
    csv files

    Expects `article_metadata.csv` provided with Neswela
    corpus as input.

    NOTE: the splits are done following Stajner and Nisioi (2018)
    https://aclanthology.org/L18-1479.pdf; 
    "We split our training, development, and test data
    disjointly based on the topic files, ensuring that 
    the sentences from the same story (regardless of 
    their complexity levels) never appear in both
    the training and test data"

    NOTE: currently no attempt is made to ensure an even
    distribut of simplified level texts in splits.
    """

    outpath = Path(csv_file).parent
    # breakpoint()
    # expects CSV with header:
    # slug,language,title,grade_level,version,filename
    df = pd.read_csv(csv_file, header=0)

    for lang in df['language'].unique():
        outfile = outpath / f'articles_metadata_{lang}_splits.csv'

        # drop all non-english articles
        df_lang = df.drop(df[df.language != lang].index)

        articles = list(df_lang['slug'].unique())

        random.shuffle(articles)

        TEST_SIZE = len(articles) // 10
        VALID_SIZE = len(articles) // 10

        test_articles = {article: 'test' for article in articles[:TEST_SIZE]}
        valid_articles = {article: 'valid' for article in articles[TEST_SIZE:TEST_SIZE+VALID_SIZE]}
        train_articles = {article: 'train' for article in articles[TEST_SIZE+VALID_SIZE:]}

        # sanity check to ensure no overlap between splits
        assert len(test_articles.keys() & valid_articles.keys()) == 0
        assert len(test_articles.keys() & train_articles.keys()) == 0
        assert len(valid_articles.keys() & train_articles.keys()) == 0

        assigned_splits = {**test_articles, **valid_articles, **train_articles}

        df_lang['split'] = df_lang['slug'].apply(lambda x: assigned_splits[x])
        print(f'*** {lang} ***')
        print(df_lang['split'].value_counts())

        df_lang.to_csv(outfile, header=True, index=False)

if __name__ == '__main__':

    meta_data = '/srv/scratch6/kew/ats/data/en/newsela_article_corpus_2016-01-29/articles_metadata.csv'
    assign_newsela_splits(meta_data)
