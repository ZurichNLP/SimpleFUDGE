#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# Author: Tannon Kew
# nohup bash run_experiments.sh train_simple_discriminator_glove > train_disc.log &

set -e
set -x # to log experiment execution

train_simple_discriminator_glove() {

    GPU=$1
    export CUDA_VISIBLE_DEVICES=$GPU

    scratch=/srv/scratch6/kew/ats
    data_dir=$scratch/data/en/newsela_article_corpus_2016-01-29/article_sents/en
    save_dir=$scratch/fudge/discriminators/newsela4_bart_glove
    model_dir=$scratch/fudge/generators/bart_large_paraNMT_filt_fr

    mkdir -p $save_dir

    echo "Running on GPU(s) $GPU"

    python main.py \
        --task simplify \
        --data_dir $data_dir \
        --save_dir $save_dir \
        --tgt_level 4 \
        --model_path_or_name $model_dir \
        --num_workers 12 \
        --lr 1e-4 \
        --batch_size 32 \
        --epochs 10 \
        --glove 'glove-wiki-gigaword-300' \
        --wandb simple_fudge
    
    echo "Finished training discrimator"
}


