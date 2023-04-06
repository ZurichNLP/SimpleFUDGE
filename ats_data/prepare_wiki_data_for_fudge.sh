#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# adapted from https://towardsdatascience.com/pre-processing-a-wikipedia-dump-for-nlp-model-training-a-write-up-3b9176fdf67
# Author: Tannon Kew
# Warning: enwiki is a huge dump, so it's best to run on a large server
# with many cpus! And be patient... (i.e. don't wait up)

set -e

SCRATCH=/srv/scratch1/kew/ats/data/en/wiki_dumps

for LG in simplewiki enwiki
do
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
    python extract_wiki_sents.py $DATA_DIR/$LG.txt $DATA_DIR/${LG}_sents.txt
    echo ""
    echo "Succesfully extracted sentences: $DATA_DIR/${LG}_sents.txt"
    echo ""

done

echo ""
echo "Aggregating and scoring all collected data..."
echo ""

# aggregate sentences from enwiki and simplewiki into a
# single csv and score them for filtering later
python aggregate_wiki_data.py \
    $SCRATCH/enwiki/enwiki_sents.txt \
    $SCRATCH/simplewiki/simplewiki_sents.txt \
    $SCRATCH/enwiki_simplewiki.csv

echo ""
echo "Done!"
echo ""
