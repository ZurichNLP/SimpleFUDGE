#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# Author: Tannon Kew
# nohup bash run_experiments.sh train_simple_discriminator_glove > train_disc.log &

set -e
# set -x # to log experiment execution
SCRATCH=/srv/scratch6/kew/ats
BASE=$(dirname "$(readlink -f "$0")")

get_seeded_random() {
  seed="$1"
  openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt \
    </dev/zero 2>/dev/null
}

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

    mkdir -p $SAVE_DIR

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
        --epoch_max_len 500 \
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


train_newsela_ablation_discriminators() {

    GPU=$1
    TGT_LEVEL="4"
    TGT_FORMAT="article_paragraphs"
    TOKENIZER="facebook/bart-large"

    DATA_DIR=$SCRATCH/data/en/newsela_article_corpus_2016-01-29/$TGT_FORMAT
    EVAL_DATA_DIR=$SCRATCH/data/en/aligned
    # INT_DATA_DIR=$SCRATCH/data/en/newsela_article_corpus_2016-01-29/$TGT_FORMAT

    export CUDA_VISIBLE_DEVICES=$GPU
    echo "Running on GPU(s) $GPU"

    for lc in 10 50 100 500 1000 5000 10000 20000 30000 40000; do
        SAVE_DIR=$SCRATCH/fudge/discriminators/newsela_abl_${lc}_l${TGT_LEVEL}_${TGT_FORMAT}
        mkdir -p $SAVE_DIR

        for level in 0 $TGT_LEVEL; do
            head -n $lc $DATA_DIR/train_${level}.txt > $SAVE_DIR/train_${level}.txt
            cp $DATA_DIR/test_${level}.txt $SAVE_DIR/test_${level}.txt
            cp $DATA_DIR/valid_${level}.txt $SAVE_DIR/valid_${level}.txt
        done

        echo ""
        echo "Training discriminator on head -n $lc instances"
        echo ""
        
        python main.py \
            --task simplify \
            --data_dir $SAVE_DIR \
            --save_dir $SAVE_DIR \
            --tgt_level $TGT_LEVEL \
            --model_path_or_name $TOKENIZER \
            --num_workers 12 \
            --lr 1e-4 \
            --batch_size 64 \
            --epochs 20 \
            --glove 'glove-wiki-gigaword-300' \
            --wandb simple_fudge 2>&1 | tee $SAVE_DIR/train.log && echo  "Finished training discrimator $SAVE_DIR"
        
        python main.py \
            --task simplify \
            --data_dir $SAVE_DIR \
            --save_dir $SAVE_DIR \
            --model_path_or_name $TOKENIZER \
            --evaluate \
            --ckpt $SAVE_DIR/model_best.pth.tar 2>&1 | tee $SAVE_DIR/eval.log && echo "Finished evaluating discrimator $SAVE_DIR"

        cond_model="newsela_abl_${lc}_l${TGT_LEVEL}_${TGT_FORMAT}"
        gen_model="bart_large_muss_mined_en"

        for split in "dev" "test"; do
            for level in 4; do
                infile_stem="newsela_manual_v0_v${level}_${split}"
                python inference.py \
                    --infile "$EVAL_DATA_DIR/$infile_stem.tsv" --outpath $SAVE_DIR \
                    --condition_model $SCRATCH/fudge/discriminators/$cond_model \
                    --generation_model $SCRATCH/fudge/generators/$gen_model \
                    --condition_lambda 1 \
                    --precondition_topk 200 \
                    --batch_size 1 \
                    --num_beams 5 --num_return_sequences 5 \
                    --repetition_penalty 1.2 && echo "Finished decoding $infile_stem.tsv with discrimator $SAVE_DIR"
                
                python simplification_evaluation.py \
                    --src_file "$EVAL_DATA_DIR/$infile_stem.tsv" \
                    --hyp_file $SAVE_DIR/$gen_model/$cond_model/$infile_stem/lambda$lambda*.txt | tee $SAVE_DIR/$gen_model/$cond_model/$infile_stem/results.csv && echo "Finished scoring generations with discrimator $SAVE_DIR"
                
            done
        done   

    done

}


# nohup bash run_experiments.sh train_simple_newsela_discriminator_on_line_parts 4 1 article_paragraphs >| newsela_l1_lineparts_training.log &
train_simple_newsela_discriminator_on_line_parts() {

    # note: must change code in data.py `line_parts = split_line(line.strip())`

    GPU=$1
    TGT_LEVEL=$2
    TGT_FORMAT=$3 # `article_sentences` or `article_paragraphs`

    [[ -z "$TGT_FORMAT" ]] && echo "Specify either `article_sentences` or `article_paragraphs`" && exit 1

    DATA_DIR=$SCRATCH/data/en/newsela_article_corpus_2016-01-29/$TGT_FORMAT
    TOKENIZER="facebook/bart-large"
    
    SAVE_DIR=$SCRATCH/fudge/discriminators/newsela-lp_l${TGT_LEVEL}_${TGT_FORMAT}

    mkdir -p $SAVE_DIR

    export CUDA_VISIBLE_DEVICES=$GPU
    echo "Running on GPU(s) $GPU"
    # batch_size used = 64 for no line parts, 256 for line parts
    # epochs = 20 for no line parts, 8 for line parts
    python main.py \
        --task simplify \
        --data_dir $DATA_DIR \
        --save_dir $SAVE_DIR \
        --tgt_level $TGT_LEVEL \
        --model_path_or_name $TOKENIZER \
        --num_workers 12 \
        --lr 1e-4 \
        --use_line_parts \
        --batch_size 256 \
        --epochs 10 \
        --glove 'glove-wiki-gigaword-300' \
        --wandb simple_fudge
    
    echo "Finished training discrimator"
}

train_newsela_ablation_discriminators_on_line_parts() {

    GPU=$1
    TGT_LEVEL="4"
    TGT_FORMAT="article_paragraphs"
    TOKENIZER="facebook/bart-large"

    DATA_DIR=$SCRATCH/data/en/newsela_article_corpus_2016-01-29/$TGT_FORMAT
    EVAL_DATA_DIR=$SCRATCH/data/en/aligned
    # INT_DATA_DIR=$SCRATCH/data/en/newsela_article_corpus_2016-01-29/$TGT_FORMAT

    export CUDA_VISIBLE_DEVICES=$GPU
    echo "Running on GPU(s) $GPU"

    for lc in 10 50 100 500 1000 5000 10000 20000 30000 40000; do
        cond_model="newsela-lp_abl_${lc}_l${TGT_LEVEL}_${TGT_FORMAT}"
        SAVE_DIR=$SCRATCH/fudge/discriminators/$cond_model
        mkdir -p $SAVE_DIR

        for level in 0 $TGT_LEVEL; do
            head -n $lc $DATA_DIR/train_${level}.txt > $SAVE_DIR/train_${level}.txt
            cp $DATA_DIR/test_${level}.txt $SAVE_DIR/test_${level}.txt
            cp $DATA_DIR/valid_${level}.txt $SAVE_DIR/valid_${level}.txt
        done

        echo ""
        echo "Training discriminator on head -n $lc instances"
        echo ""
        
        python main.py \
            --task simplify \
            --data_dir $SAVE_DIR \
            --save_dir $SAVE_DIR \
            --tgt_level $TGT_LEVEL \
            --model_path_or_name $TOKENIZER \
            --num_workers 12 \
            --lr 1e-4 \
            --use_line_parts \
            --batch_size 64 \
            --epochs 10 \
            --glove 'glove-wiki-gigaword-300' \
            --wandb simple_fudge 2>&1 | tee $SAVE_DIR/train.log && echo  "Finished training discrimator $SAVE_DIR"
        
        python main.py \
            --task simplify \
            --data_dir $SAVE_DIR \
            --save_dir $SAVE_DIR \
            --model_path_or_name $TOKENIZER \
            --evaluate \
            --ckpt $SAVE_DIR/model_best.pth.tar 2>&1 | tee $SAVE_DIR/eval.log && echo "Finished evaluating discrimator $SAVE_DIR"

        gen_model="bart_large_muss_mined_en"

        for split in "dev" "test"; do
            for level in 4; do
                infile_stem="newsela_manual_v0_v${level}_${split}"
                python inference.py \
                    --infile "$EVAL_DATA_DIR/$infile_stem.tsv" --outpath $SAVE_DIR \
                    --condition_model $SAVE_DIR \
                    --generation_model $SCRATCH/fudge/generators/$gen_model \
                    --condition_lambda 1 \
                    --precondition_topk 200 \
                    --batch_size 1 \
                    --num_beams 5 --num_return_sequences 5 \
                    --repetition_penalty 1.2 && echo "Finished decoding $infile_stem.tsv with discrimator $SAVE_DIR"
                
                python simplification_evaluation.py \
                    --src_file "$EVAL_DATA_DIR/$infile_stem.tsv" \
                    --hyp_file $SAVE_DIR/$gen_model/$cond_model/$infile_stem/lambda$lambda*.txt | tee $SAVE_DIR/$gen_model/$cond_model/$infile_stem/results.csv && echo "Finished scoring generations with discrimator $SAVE_DIR"
                
            done
        done   

    done

}

train_simple_apa_capito_discriminator() {

    GPU=$1
    TGT_LEVEL=$2
    TGT_FORMAT=$3 # `article_sentences` or `article_paragraphs`

    [[ -z "$TGT_FORMAT" ]] && echo "Specify either `article_sentences` or `article_paragraphs`" && exit 1

    DATA_DIR=$SCRATCH/data/de/apa_capito/$TGT_FORMAT
    TOKENIZER="$SCRATCH/fudge/generators/mbart/mbart_de_20k"
    
    SAVE_DIR=$SCRATCH/fudge/discriminators/apa_capito_l${TGT_LEVEL}_${TGT_FORMAT}

    mkdir -p $SAVE_DIR

    export CUDA_VISIBLE_DEVICES=$GPU
    echo "Running on GPU(s) $GPU"

    python main.py \
        --task simplify \
        --data_dir $DATA_DIR \
        --save_dir $SAVE_DIR \
        --tgt_level $TGT_LEVEL \
        --model_path_or_name $TOKENIZER \
        --num_workers 16 \
        --lr 1e-4 \
        --batch_size 32 \
        --epochs 20 \
        --glove '/srv/scratch6/kew/de_vectors.txt' #\
        # --wandb simple_fudge
    
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

    # python convert_line_aligned_to_huggingface.py $data_dir
    python convert_line_aligned_to_jsonl.py \
        --data_dir $data_dir \
        --splits "train" "test" "valid" \
        --dataset "muss"

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

finetune_bart_large_on_muss_mined_de() {

    # adapts above function for de with MBART

    GPU='3,4,5,6'
    transformers_dir=$BASE/transformers
    save_dir=$SCRATCH/fudge/generators/mbart_large_muss_mined_de
    data_dir="$SCRATCH/muss/resources/datasets/uts_de_query-577fe7eddadb30da03c2c1a2534de9a6_db-577fe7eddadb30da03c2c1a2534de9a6_topk-8_nprobe-16_density-0.6_distance-0.05_filter_ne-False_levenshtein-0.2_simplicity-0.0"

    echo "Initialising training run on GPU(s): $GPU"
    export CUDA_VISIBLE_DEVICES=$GPU

    # python convert_line_aligned_to_huggingface.py $data_dir
    python convert_line_aligned_to_jsonl.py \
        --data_dir $data_dir \
        --splits "train" "test" "valid" \
        --dataset "muss"

    python $transformers_dir/examples/pytorch/summarization/run_summarization.py \
        --model_name_or_path "/srv/scratch6/kew/ats/fudge/generators/mbart/mbart_de_20k" \
        --output_dir $save_dir --overwrite_output_dir \
        --train_file $data_dir/train.json \
        --validation_file $data_dir/valid.json \
        --test_file $data_dir/test.json \
        --lang "de_DE" --forced_bos_token "de_DE" \
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

# nohup bash run_experiments.sh finetune_bart_large_on_supervised_labeled >| newsela_supervised_finetune.log &
finetune_bart_large_on_supervised_labeled() {

    GPU="3,4"

    input_dir=$SCRATCH/data/en/aligned
    save_dir=$SCRATCH/supervised
    data_dir=$save_dir/data

    transformers_dir=$BASE/transformers

    mkdir -p $data_dir

    python convert_line_aligned_to_jsonl.py \
        --data_dir $input_dir \
        --out_dir $data_dir \
        --splits "train" "test" "dev" \
        --dataset "newsela" --label
    
    echo "Initialising training run on GPU(s): $GPU"
    export CUDA_VISIBLE_DEVICES=$GPU

    python $transformers_dir/examples/pytorch/summarization/run_summarization.py \
        --model_name_or_path "facebook/bart-large" \
        --output_dir $save_dir/bart --overwrite_output_dir \
        --train_file $data_dir/train.json \
        --validation_file $data_dir/dev.json \
        --test_file $data_dir/test.json \
        --text_column "complex" \
        --summary_column "simple" \
        --max_source_length 256 \
        --max_target_length 128 \
        --preprocessing_num_workers 16 \
        --seed 42 \
        --overwrite_cache True \
        --learning_rate 3e-05 --weight_decay 0.01 \
        --per_device_train_batch_size 8 --gradient_accumulation_steps 1 \
        --optim adamw_hf --adam_beta1 0.9 --adam_beta2 0.999 --adam_epsilon 1e-8 \
        --lr_scheduler_type polynomial --warmup_steps 500 \
        --label_smoothing_factor 0.1 --fp16 \
        --max_steps 5000 \
        --evaluation_strategy "steps" \
        --do_train --do_eval \
        --do_predict --predict_with_generate --num_beams 4 \
        --logging_steps 100 --save_steps 100 --save_total_limit 1 \
        --metric_for_best_model "rouge1" --load_best_model_at_end \
        --report_to "wandb"

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

# bash run_experiments.sh hp_search_beam 2 50 newsela-lp_l1_article_paragraphs &
hp_search_beam() {

    GPU=$1
    export CUDA_VISIBLE_DEVICES=$GPU

    max_lines=$2
    cond_model=$3 # newsela_l4_article_paragraphs
    gen_model="bart_large_muss_mined_en"
    outdir=$SCRATCH/fudge/hpsearch/$gen_model/$cond_model/beam

    mkdir -p $outdir

    echo "Running on GPU(s) $GPU"

    python hp_search.py \
        --condition_model $SCRATCH/fudge/discriminators/$cond_model \
        --generation_model $SCRATCH/fudge/generators/$gen_model \
        --outpath $outdir \
        --data_dir $SCRATCH/data/en/aligned \
        --datasets newsela_manual_v0_v1_dev newsela_manual_v0_v2_dev newsela_manual_v0_v3_dev newsela_manual_v0_v4_dev \
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


# decode_data() {

#     # Example call:
#     #   bash run_experiments.sh decode_data 2 5 newsela_l4_article_paragraphs bart_large_muss_mined_en dev

#     gpu=$1
#     export CUDA_VISIBLE_DEVICES=$gpu
    
#     lambda=$2
#     cond_model=$3
#     gen_model=$4
#     split=$5
    
#     data_dir=$SCRATCH/data/en/aligned
#     outpath=$SCRATCH/fudge/results

#     # for file in asset_test.tsv newsela_manual_v0_v4_test.tsv wiki_manual_test.tsv
#     for file in newsela_manual_v0_v1 newsela_manual_v0_v2 newsela_manual_v0_v3 newsela_manual_v0_v4 wiki_manual asset turk; do
#         # run inference
#         python inference.py \
#             --infile $data_dir/${file}_${split}.tsv --outpath $outpath \
#             --condition_model $SCRATCH/fudge/discriminators/$cond_model \
#             --generation_model $SCRATCH/fudge/generators/$gen_model \
#             --condition_lambda $lambda \
#             --precondition_topk 200 \
#             --batch_size 1 \
#             --num_beams 5 --num_return_sequences 5 \
#             --repetition_penalty 1.2

#         # run evaluation and write result to file
#         # python simplification_evaluation.py \
#         #     --src_file $data_dir/${file}_${split}.tsv \
#         #     --hyp_file $outpath/$gen_model/$cond_model/${file}_${split}/lambda$lambda*.txt | tee -a $outpath/$gen_model/$cond_model/${file}_${split}/results.csv
#     done
# }

# decode_newsela_all_levels() {

#     # Example call:
#     #   nohup bash run_experiments.sh decode_newsela_all_levels 2 8 newsela_l3_article_paragraphs bart_large_muss_mined_en dev >| decoding.dev.3.log &
#     #   nohup bash run_experiments.sh decode_newsela_all_levels 2 5 newsela_l3_article_paragraphs bart_large_muss_mined_en dev >| decoding.dev.3.log &

#     gpu=$1
#     export CUDA_VISIBLE_DEVICES=$gpu
    
#     lambda=$2
#     cond_model=$3
#     gen_model=$4
#     split=$5
    
#     data_dir=$SCRATCH/data/en/aligned
#     outpath=$SCRATCH/fudge/results

#     for level in 1 2 3 4; do
#         # run inference
#         python inference.py \
#             --infile $data_dir/newsela_manual_v0_v${level}_${split}.tsv --outpath $outpath \
#             --condition_model $SCRATCH/fudge/discriminators/$cond_model \
#             --generation_model $SCRATCH/fudge/generators/$gen_model \
#             --condition_lambda $lambda \
#             --precondition_topk 200 \
#             --batch_size 1 \
#             --num_beams 5 --num_return_sequences 5 \
#             --repetition_penalty 1.2
#     done
# }

# bash run_experiments.sh decode_newsela_all_levels_with_l4_classifier 3
decode_newsela_all_levels_with_l4_classifier() {


    gpu=$1
    export CUDA_VISIBLE_DEVICES=$gpu
    
    data_dir=$SCRATCH/data/en/aligned
    outpath=$SCRATCH/fudge/results
    gen_model="bart_large_muss_mined_en"
    
    levels=( 1 2 3 4 )
    # lambdas=( 1 2 3 4 )

    # cond_model="newsela_l4_article_paragraphs"
    # lambdas=( 2 3 5 7 ) # best on paragrahs
    
    # cond_model="newsela_l4_article_para_sents"
    # lambdas=( 0 2 6 8 ) # best on para_sents

    cond_model="newsela-lp_l4_article_para_sents"
    lambdas=() # TBD

    for split in "test" "dev"; do
        for i in "${!levels[@]}"; do
            level="${levels[i]}"
            lambda="${lambdas[i]}"
            echo ""
            echo "Newsela level: $level - Condition lambda: $lambda"
            python inference.py \
                --infile $data_dir/newsela_manual_v0_v${level}_${split}.tsv --outpath $outpath \
                --condition_model $SCRATCH/fudge/discriminators/$cond_model \
                --generation_model $SCRATCH/fudge/generators/$gen_model \
                --condition_lambda $lambda \
                --precondition_topk 200 \
                --batch_size 1 \
                --num_beams 5 --num_return_sequences 5 \
                --repetition_penalty 1.2
            echo ""
        done
    done   

}

#   bash run_experiments.sh decode_newsela_level 6 8 1 paragraphs
#   bash run_experiments.sh decode_newsela_level 5 10 2 paragraphs
#   bash run_experiments.sh decode_newsela_level 6 10 3 paragraphs
#   bash run_experiments.sh decode_newsela_level 5 7 4 paragraphs
decode_newsela_level() {
    
    gpu=$1
    export CUDA_VISIBLE_DEVICES=$gpu
    
    lambda=$2
    level=$3
    cls_type=$4

    cond_model="newsela_l${level}_article_${cls_type}"
    gen_model="bart_large_muss_mined_en"
    
    data_dir=$SCRATCH/data/en/aligned
    outpath=$SCRATCH/fudge/results

    for split in dev test; do
        # run inference
        python inference.py \
            --infile "$data_dir/newsela_manual_v0_v${level}_${split}.tsv" --outpath "$outpath" \
            --condition_model "$SCRATCH/fudge/discriminators/$cond_model" \
            --generation_model "$SCRATCH/fudge/generators/$gen_model" \
            --condition_lambda "$lambda" \
            --precondition_topk 200 \
            --batch_size 1 \
            --num_beams 5 --num_return_sequences 5 \
            --repetition_penalty 1.2
    done
}

# bash run_experiments.sh decode_newsela_level_on_line_parts 1 1 1 paragraphs &
# bash run_experiments.sh decode_newsela_level_on_line_parts 2 4 2 paragraphs &
# bash run_experiments.sh decode_newsela_level_on_line_parts 3 4 3 paragraphs &
# bash run_experiments.sh decode_newsela_level_on_line_parts 4 5 4 paragraphs &
decode_newsela_level_on_line_parts() {
    
    gpu=$1
    export CUDA_VISIBLE_DEVICES=$gpu
    
    lambda=$2
    level=$3
    cls_type=$4

    cond_model="newsela-lp_l${level}_article_${cls_type}"
    gen_model="bart_large_muss_mined_en"
    
    data_dir=$SCRATCH/data/en/aligned
    outpath=$SCRATCH/fudge/results

    for split in dev test; do
        # run inference
        python inference.py \
            --infile "$data_dir/newsela_manual_v0_v${level}_${split}.tsv" --outpath "$outpath" \
            --condition_model "$SCRATCH/fudge/discriminators/$cond_model" \
            --generation_model "$SCRATCH/fudge/generators/$gen_model" \
            --condition_lambda "$lambda" \
            --precondition_topk 200 \
            --batch_size 1 \
            --num_beams 5 --num_return_sequences 5 \
            --repetition_penalty 1.2
    done
}



# bash run_experiments.sh decode_supervised_labeled 0
decode_supervised_labeled() {

    input_dir=$SCRATCH/data/en/aligned
    exp_dir=$SCRATCH/supervised
    outpath=$exp_dir/results
    
    gpu=$1
    export CUDA_VISIBLE_DEVICES=$gpu

    for level in 1 2 3 4; do   
        for split in dev test; do
            # insert labels used in training
            cat $input_dir/newsela_manual_v0_v${level}_${split}.tsv | sed "s/^/<l${level}> /" > $exp_dir/data/newsela_manual_v0_v${level}_${split}.tsv
        
            # run inference
            # TODO update $SCRATCH/supervised/checkpoint-500 to best model
            python inference.py \
                --infile $exp_dir/data/newsela_manual_v0_v${level}_${split}.tsv --outpath $outpath \
                --generation_model $SCRATCH/supervised/bart1300 \
                --condition_lambda "0" \
                --batch_size 1 \
                --num_beams 5 --num_return_sequences 5 \

            # constructed outphath has the form:
            # echo "$outpath/$gen_model/$cond_model/newsela_manual_v0_v${level}_${split}"
            hyp_files=$(find $outpath/newsela_manual_v0_v${level}_${split} -name "lambda$lambda*.txt")
            for hyp_file in $hyp_files; do
                echo "Running evaluation on $hyp_file"
                python simplification_evaluation.py \
                    --src_file $input_dir/newsela_manual_v0_v${level}_${split}.tsv \
                    --hyp_file $hyp_file
            done
        done
    done

}



"$@"