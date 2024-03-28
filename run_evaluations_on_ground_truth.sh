#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# This script is used to evaluate the simplification quality of the ground truth data.
# Example call:
# bash run_evaluations_on_ground_truth.sh 0


BASE=$(dirname "$(readlink -f "$0")")
SCRATCH=$BASE/resources

src_files=$SCRATCH/data/en/aligned

gpu=${1:-"0"}

export CUDA_VISIBLE_DEVICES=$gpu # recommended if computing ppl with gpt2

for split in test testcattrain; do
    for level in 1 2 3 4; do
        # resources/data/en/aligned/newsela_manual_v0_v1_testcattrain.tsv
        infile=$SCRATCH/data/en/aligned/newsela_manual_v0_v${level}_${split}.tsv
        # extract 2 column from infile as hyps
        tmpfile=$(mktemp)
        cut -f2 $infile > $tmpfile
        echo "Evaluating $tmpfile against $infile"        
        # result file
        outfile=$SCRATCH/results/newsela_manual_v0_v${level}_${split}_ground_truth.csv
        # run eval
        python evaluation/simplification_evaluation_v2.py ${tmpfile} --src_file $infile --use_cuda --lens_model_path "/srv/scratch1/kew/llm_ats/LENS/checkpoints/epoch=5-step=6102.ckpt" --out_file $outfile
        # remove tmpfile
        rm $tmpfile
    done
done

