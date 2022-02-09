#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# python scripts/simplify.py scripts/examples.en --model-name muss_en_wikilarge_mined

CONDA_INIT=/home/user/kew/anaconda3/etc/profile.d/conda.sh
MUSS_DIR=/home/user/kew/INSTALLS/muss
SRC_FILE="$1"
OUT_FILE="$2"
GPU="$3"

if [ -z "$SRC_FILE" ]
  then 
    echo "No source file provided for translating!" && exit 1
fi

export CUDA_VISIBLE_DEVICES=$GPU

source $CONDA_INIT

conda activate muss

echo "Activated environment: $CONDA_DEFAULT_ENV ..."

# create temp file for cutting the source column from datasets 
tmpfile=$(mktemp /tmp/muss_src_file.XXXXXX)
echo "Translating $tmpfile ..."

# trim first column from file
cut -f 1 /srv/scratch6/kew/ats/data/en/aligned/turk_test.tsv >| $tmpfile

python $MUSS_DIR/scripts/simplify.py $tmpfile --model-name muss_en_mined --outfile $OUT_FILE

# clean up tmpfile
rm $tmpfile

conda deactivate
echo "Finished implifying with MUSS $SRC_FILE"
echo "Simplifications: $OUT_FILE"