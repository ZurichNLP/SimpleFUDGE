#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import random
import pandas as pd

SEED = 42
random.seed(SEED) # set seed for reproducibility
tgt_lang = 'en'

meta_data = '/srv/scratch6/kew/ats/data/en/newsela_article_corpus_2016-01-29/articles_metadata.csv'
outfile = '/srv/scratch6/kew/ats/data/en/newsela_article_corpus_2016-01-29/articles_metadata_{tgt_lang}_splits.csv'

# outpath = '/srv/scratch6/kew/ats/data/en/newsela_article_corpus_2016-01-29/splits/'
# if not Path(outpath).exists() and not Path(outpath).is_dir():
#     Path(outpath).mkdir(parents=True, exist_ok=False)

# Header:
# slug,language,title,grade_level,version,filename
df = pd.read_csv(meta_data, header=0)

# drop all non-english articles
df = df.drop(df[df.language != tgt_lang].index)

articles = list(df['slug'].unique())

random.shuffle(articles)

TEST_SIZE = len(articles) // 10
VALID_SIZE = len(articles) // 10

test_articles = set(articles[:TEST_SIZE])
valid_articles = set(articles[TEST_SIZE:TEST_SIZE+VALID_SIZE])
train_articles = set(articles[TEST_SIZE+VALID_SIZE:])

# sanity check to ensure no overlap between splits
assert test_articles.intersection(valid_articles) is None
assert test_articles.intersection(train_articles) is None
assert valid_articles.intersection(train_articles) is None

assigned_splits = {}

for article in test_articles:
    assigned_splits[article] = 'test'
for article in valid_articles:
    assigned_splits[article] = 'valid'
for article in train_articles:
    assigned_splits[article] = 'train'

df['split'] = df['slug'].apply(lambda x: assigned_splits[x])

breakpoint

df.to_csv(outfile, header=True, index=False)

# data_splits = {
#     'test': test_articles,
#     'valid': valid_articles,
#     'train': train_articles,
#     }

# def get_split(artcile_name, data_splits):
#     if artcile_name in data_splits['test']:
#         return 'test'
#     elif 

# for split, articles in data_splits.items():
#     outfile = Path(outpath) / f'articles.{split}.txt'
    
#     with open(outfile, 'w', encoding='utf8') as f:
#         for i, article in enumerate(articles):
#             f.write(article + '\n')

#     print(f'wrote {i} article titles to {outfile}')