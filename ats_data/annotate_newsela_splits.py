#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: Tannon Kew

"""
Updates Newsela article metadata with split informtation
(e.g. train, test, dev).

For English, splits are assigned according to articles used in Newsela-manual from
Jiang et al. 2020 (https://www.aclweb.org/anthology/2020.acl-main.709).

Expects Newsela metadata CSV with header:
    slug,language,title,grade_level,version,filename

Outputs an updated, language-specific metadata CSV file with
header:
    slug,language,title,grade_level,version,filename,split

Example call:

    python annotate_newsela_splits.py /srv/scratch6/kew/ats/data/en/newsela_article_corpus_2016-01-29
"""

import sys
from pathlib import Path
import pandas as pd

newsela_manual_dev_article_titles = [
    'marktwain-newspaper',
    'asian-modelminority',
    'return-trip',
    'sesamestreet-preschool',
    'emergencyresponse-robots',
]

newsela_manual_test_article_titles = [
    'chinook-recognition',
    'auschwitz-palestine',
    'eggprices-rising',
    'asteroid-corral',
    'timetravel-paradox',
    'alienplanet-swim',
    'airstrikes-iraq',
    'harvarddebate-versusinmates',
    'syria-refugees',
    'dinosaur-colors',
]

newsela_manual_train_article_titles = [
    'miami-searise',
    'muslim-challenges',
    'doodler-nebraskalawmaker',
    'military-police',
    'basketball-mentors',
    '3d-indoormap',
    'migrantkids-uprooted',
    'school-threats',
    'football-virtualreality',
    'antelopevalley-bomber',
    'periodictable-elements',
    'class-sizes',
    'libya-boatcapsize',
    'nativeamerican-diets',
    'solar-panels',
    'dali-vr',
    'syria-inspectors',
    'aquaponics-farm',
    'hawaii-homeless',
    'bee-deaths',
    'china-aviation',
    'google-selfcars',
    'shuttle-parts',
    'iran-water',
    'asia-ozone',
    'concussion-study',
    'koala-trees',
    'deportee-videogame',
    'pakistan-earthquake',
    'drones-wildfires',
    'digital-giving',
    'amtrak-crash',
    'botany-students',
    'comcast-merger',
    'dog-drinking',
]

def assign_splits_randomly(df, outfile, TEST_SIZE=10, VALID_SIZE=5):
    """
    assigns test/train/valid split labels to articles in the
    Newsela corpus and splits them into langauge-specific
    csv meta files

    Expects `article_metadata.csv` provided with Neswela
    corpus as input.

    The splits are done according to articles in Jiang
    et al.'s (2020) annotated datasets for "Newsela-Manual".
    
    Following Stajner and Nisioi (2018) (https://aclanthology.org/L18-1479.pdf),
    splits are disjointly based on the topic files, ensuring that 
    the sentences from the same story (regardless of 
    their complexity levels) never appear in both
    the training and test data.
    """

    articles = list(df['slug'].unique())

    test_articles = {article: 'test' for article in articles[:TEST_SIZE]}
    valid_articles = {article: 'valid' for article in articles[TEST_SIZE:TEST_SIZE+VALID_SIZE]}
    train_articles = {article: 'train' for article in articles[TEST_SIZE+VALID_SIZE:]}

    # sanity check to ensure no overlap between splits
    assert len(test_articles.keys() & valid_articles.keys()) == 0
    assert len(test_articles.keys() & train_articles.keys()) == 0
    assert len(valid_articles.keys() & train_articles.keys()) == 0

    assigned_splits = {**test_articles, **valid_articles, **train_articles}

    df['split'] = df['slug'].apply(lambda x: assigned_splits[x])
    print(f'*** {outfile} ***')
    print(df['split'].value_counts())

    df.to_csv(outfile, header=True, index=False)

def assign_splits_from_newsela_manual(df, outfile):

    def lookup(slug):
        if slug in newsela_manual_test_article_titles:
            return 'test'
        elif slug in newsela_manual_dev_article_titles:
            return 'valid'
        else:
            return 'train'

    df['split'] = df['slug'].apply(lookup)
    print(f'*** {outfile} ***')
    print(df['split'].value_counts())
    df.to_csv(outfile, header=True, index=False)


def update_meta_data(meta_data, lang='en'):
    outpath = Path(meta_data).parent
    df = pd.read_csv(meta_data, header=0)
    # drop all foreign lang articles
    df = df.drop(df[df.language != lang].index)
    outfile = outpath / f'articles_metadata_{lang}_splits.csv'
    
    if lang == 'en':
        # do english according to newsela-manual articles
        assign_splits_from_newsela_manual(df, outfile) 
    elif lang == 'es':
        assign_splits_randomly(df, outfile) # do spanish articles randomly

if __name__ == '__main__':

    orig_meta_data_file = Path(sys.argv[1]) / 'articles_metadata.csv'
    update_meta_data(orig_meta_data_file, 'en')
    update_meta_data(orig_meta_data_file, 'es')
