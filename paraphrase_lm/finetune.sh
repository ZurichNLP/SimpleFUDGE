#!/usr/bin/env bash
# -*- coding: utf-8 -*-

finetune_bart_base_on_paraNMT() {

    GPU=$1
    save_dir=/srv/scratch6/kew/paraphrase/models/
    save_pref=bart_base_paraNMT_filtered
    data_dir=/srv/scratch6/kew/paraphrase/paraNMT/paranmt_filtered/

    echo "Initialising training run on GPU(s): $GPU"
    export CUDA_VISIBLE_DEVICES=$GPU

    mkdir -p $save_dir/$save_pref/
    echo "Output directory: $save_dir/$save_pref/"
    log_file=$save_dir/$save_pref/finetune.log

    nohup python model.py --train \
        --from_pretrained facebook/bart-base \
        --save_dir $save_dir \
        --save_prefix $save_pref \
        --train_csv $data_dir/train.csv \
        --valid_csv $data_dir/dev.csv \
        --freeze_embeds --freeze_encoder \
        --batch_size 16 --grad_accum 2 \
        --max_epochs 4 --early_stopping_metric vloss \
        --warmup 500 \
        --num_workers 8 \
        --wandb paraphrasing \
        --progress_bar_refresh_rate 100 >| $log_file &
    
    if [ $? -eq 0 ]; then
        echo "Fine-tuning launched with nohup ..."
        echo "Logging to file $log_file ..."
    else
        echo "[!] fine-tuning failed"
    fi
}
}

finetune_bart_large_on_paraNMT() {

    GPU='2,3,4,5'
    save_dir=/srv/scratch6/kew/paraphrase/models
    save_pref=bart_large_paraNMT_filt_fr # frozen encoder and embeddings
    data_dir=/srv/scratch6/kew/paraphrase/paraNMT/paranmt_filtered

    echo "Initialising training run on GPU(s): $GPU"
    export CUDA_VISIBLE_DEVICES=$GPU

    mkdir -p $save_dir/$save_pref/
    echo "Output directory: $save_dir/$save_pref/"
    log_file=$save_dir/$save_pref/finetune.log

    nohup python model.py --train \
        --from_pretrained facebook/bart-large \
        --save_dir $save_dir \
        --save_prefix $save_pref \
        --train_csv $data_dir/train.csv \
        --valid_csv $data_dir/dev.csv \
        --freeze_embeds --freeze_encoder \
        --batch_size 8 --gpus 4 \
        --max_epochs 4 --early_stopping_metric vloss \
        --warmup 500 \
        --num_workers 8 \
        --wandb paraphrasing \
        --progress_bar_refresh_rate 100 >| $log_file &
    
    if [ $? -eq 0 ]; then
        echo "Fine-tuning launched with nohup ..."
        echo "Logging to file $log_file ..."
    else
        echo "[!] fine-tuning failed"
    fi
}

"$@"