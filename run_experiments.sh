#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# Author: Tannon Kew
# nohup bash run_experiments.sh train_simple_discriminator_glove > train_disc.log &

set -e
# set -x # to log experiment execution
SCRATCH=/srv/scratch6/kew/ats

demo() {

    GPU=$1
    export CUDA_VISIBLE_DEVICES=$GPU
    cond_model=newsela4_bart_glove_bi
    # cond_model=newsela4_bart_glove
    # cond_model=wiki100M_bart_glove
    gen_model=bart_large_paraNMT_filt_fr
    lambda=5

    python predict_simplify.py \
        --condition_model $SCRATCH/fudge/discriminators/$cond_model \
        --generation_model $SCRATCH/fudge/generators/$gen_model \
        --condition_lambda $lambda \
        --num_beams 1 --num_return_sequences 1 \
        --analysis_file /srv/scratch6/kew/ats/fudge/analysis/newsela4_bart_glove-lambda$lambda.json \
        --input_text "Memorial West's class is one of several programs offered through hospitals to help children stay healthy through exercise and proper eating"
        # --do_sample True --typical_p 0.5
        
}

train_simple_newsela_discriminator_glove() {

    GPU=$1
    export CUDA_VISIBLE_DEVICES=$GPU

    data_dir=$SCRATCH/data/en/newsela_article_corpus_2016-01-29/article_sents
    save_dir=$SCRATCH/fudge/discriminators/newsela4_bart_glove
    model_dir=$SCRATCH/fudge/generators/bart_large_paraNMT_filt_fr

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

train_simple_newsela_discriminator_glove_bidirectional() {

    GPU=$1
    export CUDA_VISIBLE_DEVICES=$GPU

    data_dir=$SCRATCH/data/en/newsela_article_corpus_2016-01-29/article_sents
    save_dir=$SCRATCH/fudge/discriminators/newsela4_bart_glove_bi
    model_dir=$SCRATCH/fudge/generators/bart_large_paraNMT_filt_fr

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
        --wandb simple_fudge \
        --bidirectional True
    
    echo "Finished training discrimator"
}

train_simple_newsela2_discriminator_glove_bidirectional() {

    GPU=$1
    export CUDA_VISIBLE_DEVICES=$GPU

    data_dir=$SCRATCH/data/en/newsela_article_corpus_2016-01-29/article_sents
    save_dir=$SCRATCH/fudge/discriminators/newsela2_bart_glove_bi
    model_dir=$SCRATCH/fudge/generators/bart_large_paraNMT_filt_fr

    mkdir -p $save_dir

    echo "Running on GPU(s) $GPU"

    python main.py \
        --task simplify \
        --data_dir $data_dir \
        --save_dir $save_dir \
        --tgt_level 2 \
        --model_path_or_name $model_dir \
        --num_workers 12 \
        --lr 1e-4 \
        --batch_size 32 \
        --epochs 10 \
        --glove 'glove-wiki-gigaword-300' \
        --wandb simple_fudge \
        --bidirectional True
    
    echo "Finished training discrimator"
}

train_simple_wiki_discriminator_glove() {

    GPU=$1
    export CUDA_VISIBLE_DEVICES=$GPU

    data_dir=$SCRATCH/data/en/wiki_dumps
    save_dir=$SCRATCH/fudge/discriminators/wiki100M_bart_glove
    model_dir=$SCRATCH/fudge/generators/bart_large_paraNMT_filt_fr

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

hp_search_test() {

    GPU=$1
    export CUDA_VISIBLE_DEVICES=$GPU

    cond_model=wiki100M_bart_glove
    gen_model=bart_large_paraNMT_filt_fr
    outdir=$SCRATCH/fudge/hpsearch/scratch

    mkdir -p $outdir

    echo "Running on GPU(s) $GPU"

    python hp_search.py \
        --condition_model $SCRATCH/fudge/discriminators/$cond_model \
        --generation_model $SCRATCH/fudge/generators/$gen_model \
        --outpath $outdir \
        --data_dir $SCRATCH/data/en/aligned \
        --datasets asset_validation newsela_manual_v0_v4_dev \
        --max_lines 10

    echo "Finished HP sweep. See results in $outdir"
}

hp_search_beam() {

    GPU=$1
    export CUDA_VISIBLE_DEVICES=$GPU

    max_lines=$2
    # cond_model=wiki100M_bart_glove
    cond_model=$3 # newsela4_bart_glove_bi
    # cond_model=newsela4_bart_glove
    # cond_model=wiki100M_bart_glove
    gen_model=$4 # bart_large_paraNMT_filt_fr
    # gen_model=bart_large_paraNMT_filt_fr
    outdir=$SCRATCH/fudge/hpsearch/$gen_model/$cond_model/beam

    mkdir -p $outdir

    echo "Running on GPU(s) $GPU"

    python hp_search.py \
        --condition_model $SCRATCH/fudge/discriminators/$cond_model \
        --generation_model $SCRATCH/fudge/generators/$gen_model \
        --outpath $outdir \
        --data_dir $SCRATCH/data/en/aligned \
        --datasets asset_validation turk_validation newsela_manual_v0_v4_dev wiki_manual_dev \
        --max_lines $max_lines --batch_size 1 \
        --log_to_file

    echo "Finished HP sweep. See results in $outdir"
}

hp_search_topk() {

    GPU=$1
    export CUDA_VISIBLE_DEVICES=$GPU

    cond_model=wiki100M_bart_glove
    gen_model=bart_large_paraNMT_filt_fr
    outdir=$SCRATCH/fudge/hpsearch/$gen_model/$cond_model/topk5

    mkdir -p $outdir

    echo "Running on GPU(s) $GPU"

    python hp_search.py \
        --condition_model $SCRATCH/fudge/discriminators/$cond_model \
        --generation_model $SCRATCH/fudge/generators/$gen_model \
        --do_sample True --top_k 5 \
        --outpath $outdir \
        --data_dir $SCRATCH/data/en/aligned \
        --datasets asset_validation turk_validation newsela_manual_v0_v4_dev wiki_manual_dev \
        --max_lines 50 \
        --log_to_file

    echo "Finished HP sweep. See results in $outdir"
}


decode_data() {

    GPU=$1
    export CUDA_VISIBLE_DEVICES=$GPU
    
    condition_lambda=$2
    cond_model=$3 # newsela4_bart_glove_bi
    # cond_model=newsela4_bart_glove
    # cond_model=wiki100M_bart_glove
    gen_model=$4 # bart_large_paraNMT_filt_fr
    data_dir=$SCRATCH/data/en/aligned
    outpath=$SCRATCH/fudge/results

    # for file in asset_test.tsv newsela_manual_v0_v4_test.tsv wiki_manual_test.tsv
    for file in newsela_manual_v0_v4_dev.tsv wiki_manual_dev.tsv asset_validation.tsv
    do 
        python inference.py \
            --infile $data_dir/$file --outpath $outpath \
            --condition_model $SCRATCH/fudge/discriminators/$cond_model \
            --generation_model $SCRATCH/fudge/generators/$gen_model \
            --condition_lambda $condition_lambda \
            --precondition_topk 100 \
            --batch_size 1 \
            --num_beams 5 --num_return_sequences 5
    done
}


"$@"