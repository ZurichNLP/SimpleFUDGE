#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# Author: Tannon Kew
# nohup bash run_experiments.sh train_simple_discriminator_glove > train_disc.log &

set -e
# set -x # to log experiment execution

train_simple_newsela_discriminator_glove() {

    GPU=$1
    export CUDA_VISIBLE_DEVICES=$GPU

    scratch=/srv/scratch6/kew/ats
    data_dir=$scratch/data/en/newsela_article_corpus_2016-01-29/article_sents
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


train_simple_wiki_discriminator_glove() {

    GPU=$1
    export CUDA_VISIBLE_DEVICES=$GPU

    scratch=/srv/scratch6/kew/ats
    data_dir=$scratch/data/en/wiki_dumps
    save_dir=$scratch/fudge/discriminators/wiki100M_bart_glove
    model_dir=$scratch/fudge/generators/bart_large_paraNMT_filt_fr

    mkdir -p $save_dir

    echo "Running on GPU(s) $GPU"

    python main.py \
        --task simplify \
        --data_dir $data_dir \
        --save_dir $save_dir \
        --model_path_or_name $model_dir \
        --num_workers 12 \
        --lr 1e-4 \
        --batch_size 128 \
        --epochs 12 \
        --epoch_max_len 500000 \
        --glove 'glove-wiki-gigaword-300' \
        --wandb simple_fudge
    
    echo "Finished training discrimator"
}


hp_search_beam() {

    GPU=$1
    export CUDA_VISIBLE_DEVICES=$GPU

    scratch=/srv/scratch6/kew/ats
    cond_model=wiki100M_bart_glove
    gen_model=bart_large_paraNMT_filt_fr
    outdir=$scratch/fudge/hpsearch/$gen_model-$cond_model

    mkdir -p $outdir

    echo "Running on GPU(s) $GPU"

    python hp_search.py \
        --condition_model $scratch/fudge/discriminators/$cond_model/model_best.pth.tar \
        --generation_model $scratch/fudge/generators/$gen_model \
        --outpath $outdir \
        --data_dir $scratch/data/en/aligned \
        --datasets asset_validation turk_validation newsela_manual_v0_v4_dev wiki_manual_dev \
        --max_lines 50

    echo "Finished HP sweep. See results in $outdir"
}

"$@"