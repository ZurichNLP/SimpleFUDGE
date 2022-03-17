#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# Author: Tannon Kew
# nohup bash run_experiments.sh train_simple_discriminator_glove > train_disc.log &

set -e
# set -x # to log experiment execution
SCRATCH=/srv/scratch6/kew/ats
BASE=$(dirname "$(readlink -f "$0")")

##############################
# FUDGE DISCRIMINATOR TRAINING
##############################

train_simple_newsela_discriminator_glove() {

    GPU=$1
    export CUDA_VISIBLE_DEVICES=$GPU

    data_dir=$SCRATCH/data/en/newsela_article_corpus_2016-01-29/article_sentences
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

    data_dir=$SCRATCH/data/en/newsela_article_corpus_2016-01-29/article_sentences
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

train_simple_wiki_discriminator() {

    GPU=$1
    export CUDA_VISIBLE_DEVICES=$GPU

    DATA_DIR=$SCRATCH/data/en/wiki_dumps
    SAVE_DIR=$SCRATCH/fudge/discriminators/wiki100M_bart_glove
    TOKENIZER="facebook/bart-large"

    mkdir -p $save_dir

    echo "Running on GPU(s) $GPU"

    python main.py \
        --task simplify \
        --data_dir $DATA_DIR \
        --save_dir $SAVE_DIR \
        --model_path_or_name $TOKENIZER \
        --num_workers 12 \
        --lr 1e-4 \
        --batch_size 128 \
        --epochs 12 \
        --epoch_max_len 500000 \
        --glove 'glove-wiki-gigaword-300' \
        --wandb simple_fudge
    
    echo "Finished training discrimator"
}

train_simple_newsela_discriminator() {

    GPU=$1
    TGT_LEVEL=$2
    TGT_FORMAT=$3 # `article_sentences` or `article_paragraphs`

    [[ -z "$TGT_FORMAT" ]] && echo "Specify either `article_sentences` or `article_paragraphs`" && exit 1

    DATA_DIR=$SCRATCH/data/en/newsela_article_corpus_2016-01-29/$TGT_FORMAT
    TOKENIZER="facebook/bart-large"
    
    SAVE_DIR=$SCRATCH/fudge/discriminators/newsela_l${TGT_LEVEL}_${TGT_FORMAT}

    mkdir -p $SAVE_DIR

    export CUDA_VISIBLE_DEVICES=$GPU
    echo "Running on GPU(s) $GPU"

    python main.py \
        --task simplify \
        --data_dir $DATA_DIR \
        --save_dir $SAVE_DIR \
        --tgt_level $TGT_LEVEL \
        --model_path_or_name $TOKENIZER \
        --num_workers 12 \
        --lr 1e-4 \
        --batch_size 64 \
        --epochs 20 \
        --glove 'glove-wiki-gigaword-300' \
        --wandb simple_fudge
    
    echo "Finished training discrimator"
}

##############################
# PARAPHRASE MODEL FINETUNEING
##############################

finetune_bart_large_on_muss_mined() {

    GPU='3,4,5,6'
    transformers_dir=$BASE/transformers
    save_dir=$SCRATCH/fudge/generators/bart_large_muss_mined_en
    data_dir=$SCRATCH/muss/resources/datasets/muss_mined_paraphrases/en_mined_paraphrases

    echo "Initialising training run on GPU(s): $GPU"
    export CUDA_VISIBLE_DEVICES=$GPU

    python $transformers_dir/examples/pytorch/summarization/run_summarization.py \
        --model_name_or_path "facebook/bart-large" \
        --output_dir $save_dir --overwrite_output_dir \
        --train_file $data_dir/train.json \
        --validation_file $data_dir/valid.json \
        --test_file $data_dir/test.json \
        --text_column "complex" \
        --summary_column "simple" \
        --max_source_length 1024 \
        --max_target_length 256 \
        --preprocessing_num_workers 16 \
        --seed 42 \
        --overwrite_cache True \
        --learning_rate 3e-05 --weight_decay 0.01 \
        --per_device_train_batch_size 8 --gradient_accumulation_steps 2 \
        --optim adamw_hf --adam_beta1 0.9 --adam_beta2 0.999 --adam_epsilon 1e-8 \
        --lr_scheduler_type polynomial --warmup_steps 500 \
        --label_smoothing_factor 0.1 --fp16 \
        --max_steps 20000 \
        --evaluation_strategy "steps" \
        --do_train --do_eval \
        --do_predict --num_beams 4 --prediction_loss_only \
        --logging_steps 100 --save_steps 100 --save_total_limit 1 \
        --metric_for_best_model "loss" --load_best_model_at_end \
        --report_to "wandb"
    
    wait

    if [ $? -eq 0 ]; then
        echo "Fine-tuning finished successfully"
    else
        echo "[!] fine-tuning failed"
    fi

}

###########
# HP SEARCH
###########

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
        --datasets asset_dev newsela_manual_v0_v4_dev \
        --max_lines 10

    echo "Finished HP sweep. See results in $outdir"
}

hp_search_beam() {

    GPU=$1
    export CUDA_VISIBLE_DEVICES=$GPU

    max_lines=$2
    cond_model=$3 # newsela_l4_article_paragraphs
    gen_model=$4 # bart_large_muss_mined_en
    outdir=$SCRATCH/fudge/hpsearch/$gen_model/$cond_model/beam

    mkdir -p $outdir

    echo "Running on GPU(s) $GPU"

    python hp_search.py \
        --condition_model $SCRATCH/fudge/discriminators/$cond_model \
        --generation_model $SCRATCH/fudge/generators/$gen_model \
        --outpath $outdir \
        --data_dir $SCRATCH/data/en/aligned \
        --datasets newsela_manual_v0_v1_dev newsela_manual_v0_v2_dev newsela_manual_v0_v3_dev newsela_manual_v0_v4_dev asset_dev turk_dev wiki_manual_dev \
        --max_lines $max_lines --batch_size 1 \
        --log_to_file

    echo "Finished HP sweep. See results in $outdir"
}

hp_search_topk() {

    GPU=$1
    export CUDA_VISIBLE_DEVICES=$GPU

    cond_model=newsela_l4_article_paragraphs
    gen_model=bart_large_muss_mined_en
    outdir=$SCRATCH/fudge/hpsearch/$gen_model/$cond_model/topk5

    mkdir -p $outdir

    echo "Running on GPU(s) $GPU"

    python hp_search.py \
        --condition_model $SCRATCH/fudge/discriminators/$cond_model \
        --generation_model $SCRATCH/fudge/generators/$gen_model \
        --do_sample True --top_k 5 \
        --outpath $outdir \
        --data_dir $SCRATCH/data/en/aligned \
        --datasets newsela_manual_v0_v4_dev asset_dev turk_dev wiki_manual_dev \
        --max_lines 50 --batch_size 1 \
        --log_to_file

    echo "Finished HP sweep. See results in $outdir"
}

########################
# GENERATION / INFERENCE
########################

demo() {

    GPU=$1
    export CUDA_VISIBLE_DEVICES=$GPU
    cond_model=newsela_l4_article_paragraphs
    gen_model=bart_large_muss_mined_en
    lambda=5

    python predict_simplify.py \
        --condition_model $SCRATCH/fudge/discriminators/$cond_model \
        --generation_model $SCRATCH/fudge/generators/$gen_model \
        --condition_lambda $lambda \
        --num_beams 1 --num_return_sequences 1 \
        --input_text "Memorial West's class is one of several programs offered through hospitals to help children stay healthy through exercise and proper eating"
        
        # --analysis_file $SCRATCH//fudge/analysis/${gen_model}-${cond_model}-l${lambda}.json \
        # --do_sample True --typical_p 0.5
        
}


decode_data() {

    # Example call:
    #   bash run_experiments.sh decode_data 2 5 newsela_l4_article_paragraphs bart_large_paraNMT_filt_fr dev

    gpu=$1
    export CUDA_VISIBLE_DEVICES=$gpu
    
    lambda=$2
    cond_model=$3
    gen_model=$4
    split=$5
    
    data_dir=$SCRATCH/data/en/aligned
    outpath=$SCRATCH/fudge/results

    # for file in asset_test.tsv newsela_manual_v0_v4_test.tsv wiki_manual_test.tsv
    for file in newsela_manual_v0_v1 newsela_manual_v0_v2 newsela_manual_v0_v3 newsela_manual_v0_v4 wiki_manual asset
    do
        # run inference
        python inference.py \
            --infile $data_dir/${file}_${split}.tsv --outpath $outpath \
            --condition_model $SCRATCH/fudge/discriminators/$cond_model \
            --generation_model $SCRATCH/fudge/generators/$gen_model \
            --condition_lambda $lambda \
            --precondition_topk 100 \
            --batch_size 1 \
            --num_beams 5 --num_return_sequences 5
        # run evaluation and write result to file
        python simplification_evaluation.py \
            --src_file $data_dir/${file}_${split}.tsv \
            --hyp_file $outpath/$gen_model/$cond_model/${file}_${split}/lambda$lambda*.txt | tee -a $outpath/$gen_model/$cond_model/${file}_${split}/results.csv
    done
}



"$@"