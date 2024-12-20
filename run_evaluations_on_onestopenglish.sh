#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# Example call:
# bash run_evaluations_on_onestopenglish.sh 0 test

# Wraps call to simplification evaluation for all relevant outputs in one script, e.g.
# python evaluation/simplification_evaluation_v2.py \
#     resources/muss/outputs/onestopenglish_l0_l1_dev*.pred \
#     --src_file resources/data/en/aligned/onestopenglish_l0_l1_dev.tsv \

BASE=$(dirname "$(readlink -f "$0")")
SCRATCH=$BASE/resources

src_files=$SCRATCH/data/en/aligned
muss_outputs=$SCRATCH/muss/outputs
super_outputs=$SCRATCH/supervised
fudge_outputs=$SCRATCH/fudge/outputs/bart_large_muss_mined_en
fsllm_outputs=$SCRATCH/fsllm/outputs

# init results as header from evaldataframe
results=$"bleu;sari;sari_add;sari_keep;sari_del;fkgl;pbert_ref;rbert_ref;fbert_ref;pbert_src;rbert_src;fbert_src;ppl_mean;ppl_std;lens;lens_std;intra_dist1;intra_dist2;inter_dist1;inter_dist2;Compression ratio;Sentence splits;Levenshtein similarity;Exact copies;Additions proportion;Deletions proportion;Lexical complexity score;file_id\n"

gpu=${1:-"0"}
split=${2:-"test"}

export CUDA_VISIBLE_DEVICES=$gpu # recommended if computing ppl with gpt2

# ground truth evaluations
for level in 1 2; do
    infile=$SCRATCH/data/en/aligned/onestopenglish_l0_l${level}_${split}.tsv
    # extract 2 column from infile as hyps
    tmpfile=$(mktemp)
    cut -f2 $infile > $tmpfile
    echo "Evaluating $tmpfile against $infile"        
    # result file
    outfile=$SCRATCH/results/onestopenglish_l0_l${level}_${split}_ground_truth.csv
    # run eval
    python evaluation/simplification_evaluation_v2.py ${tmpfile} --src_file $infile --use_cuda --lens_model_path "/srv/scratch1/kew/llm_ats/LENS/checkpoints/epoch=5-step=6102.ckpt" --out_file $outfile
    # remove tmpfile
    rm $tmpfile
done


# # # evaluate muss outputs on onestopenglish
tgt_dir=$muss_outputs
echo ""
echo "***** Current target dir: $tgt_dir *****"
echo ""
infiles=$(find $tgt_dir -name "onestopenglish_l0_l*_${split}_*.pred")
for infile in $infiles; do
    # extract level id from filename
    level=$(echo $infile | grep -oP '(?<=l0_l)[0-9]+')
    
    echo "scoring ${infile} with level ${level} ..."
    res=$(python evaluation/simplification_evaluation_v2.py ${infile} --src_file $src_files/onestopenglish_l0_l${level}_${split}.tsv --use_cuda --lens_model_path "/srv/scratch1/kew/llm_ats/LENS/checkpoints/epoch=5-step=6102.ckpt")
    [[ ! -z "$res" ]] && results+=$"${res:301}\n" || echo "failed to score ${infile}"
done

# supervised outputs
for train_data in "onestopenglish"; do
    tgt_dir="${super_outputs}/${train_data}/outputs/*${split}"
    echo ""
    echo "***** Current target dir: $tgt_dir *****"
    echo ""

    # get all relevant files in target output dir (should be only one, but eh)
    infiles=$(find $tgt_dir -name "lambda*.txt")
    for infile in $infiles; do
        # if the relevant file exists, run eval
        level=$(echo $infile | grep -oP '(?<=l0_l)[0-9]+')

        echo "scoring ${infile} with level ${level} ..."
        res=$(python evaluation/simplification_evaluation_v2.py ${infile} --src_file $src_files/onestopenglish_l0_l${level}_${split}.tsv --use_cuda --lens_model_path "/srv/scratch1/kew/llm_ats/LENS/checkpoints/epoch=5-step=6102.ckpt")
        [[ ! -z "$res" ]] && results+=$"${res:301}\n" || echo "failed to score ${infile}"
    done
done

# evaluate fudge outputs on onestopenglish
tgt_dir="${fudge_outputs}/onestopenglish_l*/*${split}"
echo ""
echo "***** Current target dir: $tgt_dir *****"
echo ""
# get all relevant files in target output dir
infiles=$(find $tgt_dir -name "lambda*.txt")
for infile in $infiles; do
    # extract level id from filename
    level=$(echo $infile | grep -oP '(?<=l0_l)[0-9]+')

    echo "scoring ${infile} with level ${level} ..."
    res=$(python evaluation/simplification_evaluation_v2.py ${infile} --src_file $src_files/onestopenglish_l0_l${level}_${split}.tsv --use_cuda --lens_model_path "/srv/scratch1/kew/llm_ats/LENS/checkpoints/epoch=5-step=6102.ckpt")
    [[ ! -z "$res" ]] && results+=$"${res:301}\n" || echo "failed to score ${infile}"
done


# evaluate fsllm outputs on onestopenglish
for model in "TinyLlama_1.1B_intermediate_step_1431k_3T" "Llama_2_7b_hf"; do
    tgt_dir="$fsllm_outputs/$model"
    echo ""
    echo "***** Current target dir: $tgt_dir *****"
    echo ""
    
    # get all relevant files in target output dir
    infiles=$(find $tgt_dir -name "onestopenglish_l0_l*${split}-*.jsonl")
    for infile in $infiles; do
        # extract level id from filename
        level=$(echo $infile | grep -oP '(?<=l0_l)[0-9]+')

        echo "scoring ${infile} with level ${level} ..."
        res=$(python evaluation/simplification_evaluation_v2.py ${infile} --use_cuda --lens_model_path "/srv/scratch1/kew/llm_ats/LENS/checkpoints/epoch=5-step=6102.ckpt")
        # if the resul is not empyt, add to full results
        [[ ! -z "$res" ]] && results+=$"${res:301}\n" || echo "failed to score ${infile}"
    done
done


echo ""
echo "***** RESULTS *****"
echo -e $results | tee "resources/results/onestopenglish_${split}.csv"

echo "done"

