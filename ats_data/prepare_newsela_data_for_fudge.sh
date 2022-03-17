#!/usr/bin/env bash
# -*- coding: utf-8 -*-

set -e

DATA_DIR=/srv/scratch6/kew/ats/data
NEWSELA_DIR=$DATA_DIR/en/newsela_article_corpus_2016-01-29

# collect split information according to newsela manual
# Jiang et al (2021) https://arxiv.org/pdf/2005.02324.pdf
python annotate_newsela_splits.py $NEWSELA_DIR

echo ""
echo "Extracting sentences from Newsela levels..."
echo ""

# extract sentences/paragraphs for training FUDGE discriminators
# for english
python extract_newsela_data_for_fudge.py \
    --indir $NEWSELA_DIR \
    --outdir $NEWSELA_DIR/article_paragraphs \
    --meta_data $NEWSELA_DIR/articles_metadata_en_splits.csv \
    --format paragraph 

python extract_newsela_data_for_fudge.py \
    --indir $NEWSELA_DIR \
    --outdir $NEWSELA_DIR/article_sentences \
    --meta_data $NEWSELA_DIR/articles_metadata_en_splits.csv \
    --format sentence 

# for spanish
python extract_newsela_data_for_fudge.py \
    --indir $NEWSELA_DIR \
    --outdir $NEWSELA_DIR/article_paragraphs_es \
    --meta_data $NEWSELA_DIR/articles_metadata_es_splits.csv \
    --format paragraph

python extract_newsela_data_for_fudge.py \
    --indir $NEWSELA_DIR \
    --outdir $NEWSELA_DIR/article_sentences_es \
    --meta_data $NEWSELA_DIR/articles_metadata_es_splits.csv \
    --format sentence

echo ""
echo "Succesfully extracted sentences: $NEWSELA_DIR/article_sentences/"
echo ""

# extract alignments from newsela manual train/test/dev splits
for level in 1 2 3 4; do
    for split in train test dev; do
        echo "extracting aligned sentences for newsela_manual_v0_v${level}_$split ..."
        python extract_aligned_sents_wiki_newsela_manual.py \
            --infile $DATA_DIR/en/newsela-auto/newsela-manual/all/$split.tsv \
            --outfile $DATA_DIR/en/aligned/newsela_manual_v0_v${level}_$split.tsv \
            --complex_level 0 --simple_level $level
    done
done

echo "Done!"