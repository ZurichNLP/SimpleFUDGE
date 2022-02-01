#!/usr/bin/env bash
# -*- coding: utf-8 -*-

set -e

DATADIR=/srv/scratch6/kew/ats/data/en/newsela_article_corpus_2016-01-29

python get_newsela_splits.py $DATADIR/articles_metadata.csv

echo ""
echo "Extracting sentences from Newsela levels..."
echo ""

python extract_newsela_sents.py \
    --indir $DATADIR \
    --outdir $DATADIR/article_sents \
    --lang en

echo ""
echo "Succesfully extracted sentences: $DATADIR/article_sents/"
echo ""

