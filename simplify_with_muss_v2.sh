#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# Server: rattle
# bash simplify_with_muss_v2.sh 3


CONDA_INIT=/home/user/kew/anaconda3/etc/profile.d/conda.sh
# expects muss to be installed in the `installs`` directory in simple_fudge/
MUSS_DIR=installs/muss

src_data=resources/data/en/aligned/
out_data=resources/muss/outputs/
gpu=$1

# setup env
export CUDA_VISIBLE_DEVICES=$gpu
source $CONDA_INIT
conda activate muss
echo "Activated environment: $CONDA_DEFAULT_ENV ..."

# run simplification model
for level in 1 2 3 4; do
    for split in dev test; do
        python $MUSS_DIR/scripts/simplify_file.py \
            $src_data/newsela_manual_v0_v${level}_dev.tsv \
            --model-name muss_en_mined \
            --out_path $out_data \
            --params_file $out_data/finetune_newsela_v0_v${level}_50_params.json

            echo "Finised $src_data/newsela_manual_v0_v${level}_${split}.tsv ..."
    done
done

conda deactivate

