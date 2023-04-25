#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# This script prepares the Newsela data for FUDGE.
# 1) it extracts all sentences from the Newsela articles for training, testing and dev of FUDGE discriminators.
# 2) it extracts the aligned sentences from the Newsela manual train/test/dev splits for simple-FUDGE experiments.

set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
BASE=$SCRIPT_DIR/../

DATA_DIR=resources/data
NEWSELA_DIR=$DATA_DIR/en/newsela_article_corpus_2016-01-29

# collect split information according to newsela manual
# Jiang et al (2021) https://arxiv.org/pdf/2005.02324.pdf
python annotate_newsela_splits.py $NEWSELA_DIR

echo ""
echo "Extracting sentences from Newsela levels..."
echo ""

# extract sentences/paragraphs for training FUDGE discriminators
# for english
python $SCRIPT_DIR/extract_newsela_texts.py \
    --indir $NEWSELA_DIR \
    --outdir $NEWSELA_DIR/article_paragraphs \
    --meta_data newsela_articles_metadata_with_splits \
    --format paragraph 

python $SCRIPT_DIR/extract_newsela_texts.py \
    --indir $NEWSELA_DIR \
    --outdir $NEWSELA_DIR/article_sentences \
    --meta_data newsela_articles_metadata_with_splits \
    --format sentence

python $SCRIPT_DIR/extract_newsela_texts.py \
    --indir $NEWSELA_DIR \
    --outdir $NEWSELA_DIR/article_para_sents \
    --meta_data newsela_articles_metadata_with_splits \
    --format para_sent

echo ""
echo "Succesfully extracted sentences: $NEWSELA_DIR/article_sentences/"
echo ""

# extract alignments from newsela manual train/test/dev splits
for level in 1 2 3 4; do
    for split in train test dev; do
        echo "extracting aligned sentences for newsela_manual_v0_v${level}_$split ..."
        python $SCRIPT_DIR/extract_alignments_newsela.py \
            --infile $DATA_DIR/en/newsela-auto/newsela-manual/all/$split.tsv \
            --corpus_dir $NEWSELA_DIR \
            --output_dir $DATA_DIR/en/aligned/ \
            --unit sent --complex_level 0 --simple_level $level
    done
done

echo "Done!"