#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# adapted from https://towardsdatascience.com/pre-processing-a-wikipedia-dump-for-nlp-model-training-a-write-up-3b9176fdf67

set -e

# # download en wiki dump to data directory
# wget https://dumps.wikimedia.org/enwiktionary/20220120/enwiktionary-20220120-pages-articles-multistream.xml.bz2 -P $data_dir

# # download simple en wiki dump to data directory
# wget https://dumps.wikimedia.org/simplewiki/20220120/simplewiki-20220120-pages-articles-multistream.xml.bz2 -P $data_dir

LG=$1
SCRATCH=/srv/scratch6/kew/ats/data/en
DATA_DIR=$SCRATCH/$LG
WIKI_DUMP=${LG}-latest-pages-articles.xml.bz2
WIKI_DUMP_URL=https://dumps.wikimedia.org/${LG}/latest/$WIKI_DUMP

mkdir -p $DATA_DIR

# download latest Wikipedia dump in chosen language
echo "Downloading the latest $LG-language Wikipedia dump from $WIKI_DUMP_DOWNLOAD_URL..."
wget -c $WIKI_DUMP_URL -P $DATA_DIR
echo "Succesfully downloaded the latest $LG-language Wikipedia dump to $DATA_DIR/$WIKI_DUMP"

# extract and clean the chosen Wikipedia dump
# echo ""
# echo "Extracting and cleaning $DATA_DIR/$WIKI_DUMP to $OUT_DIR ..."
# echo ""
# python3 -m wikiextractor.WikiExtractor $DATA_DIR/$WIKI_DUMP \
# --processes 64 \
# -o $OUT_DIR

# extract and clean the chosen Wikipedia dump
# NOTE: we output to a single file, removing document
# boundaries and meta data
echo ""
echo "Extracting and cleaning ..."
echo ""
python3 -m wikiextractor.WikiExtractor $DATA_DIR/$WIKI_DUMP \
--processes 79 \
-q -o - \
| sed "/^\s*\$/d" \
| grep -v "^<doc id=" \
| grep -v "</doc>\$" \
> $DATA_DIR/$LG.txt

echo ""
echo "Succesfully extracted and cleaned: $DATA_DIR/$LG.txt"
echo ""


echo ""
echo "Extracting sentences from ..."
echo ""
# get sentences
python extract_sentences_from_wiki_corpus.py $DATA_DIR/$LG.txt $DATA_DIR/${LG}_sents.txt
echo ""
echo "Succesfully extracted sentences: $DATA_DIR/${LG}_sents.txt"
echo ""
