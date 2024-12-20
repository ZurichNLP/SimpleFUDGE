#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# note to self: run in llm_hf1 env

# Example call:
# bash run_evaluations_on_newsela.sh.sh 0 test
# bash run_evaluations_on_newsela.sh.sh 0 testcattrain

# Wraps call to simplification evaluation for all relevant outputs in one script, e.g.
# python evaluation/simplification_evaluation_v2.py \
#     resources/muss/outputs/newsela_manual_v0_v1_dev*.pred \
#     --src_file resources/data/en/aligned/newsela_manual_v0_v1_dev.tsv \

BASE=$(dirname "$(readlink -f "$0")")
SCRATCH=$BASE/resources

src_files=$SCRATCH/data/en/aligned
muss_outputs=$SCRATCH/muss/outputs
super_outputs=$SCRATCH/supervised
fudge_outputs=$SCRATCH/fudge/outputs/bart_large_muss_mined_en
fsllm_outputs=$SCRATCH/fsllm/outputs
# outfile=${1:"results.csv"}

# init results as header from evaldataframe
# results=$"file;ppl_diff;bleu;sari;fkgl;bertscore_p;bertscore_r;bertscore_f1;Compression ratio;Sentence splits;Levenshtein similarity;Exact copies;Additions proportion;Deletions proportion;Lexical complexity score\n"
# results=$"file;ppl;bleu;sari;fkgl;bertscore_p_ref;bertscore_r_ref;bertscore_f1_ref;bertscore_p_src;bertscore_r_src;bertscore_f1_src;intra_dist1;intra_dist2;inter_dist1;inter_dist2;Compression ratio;Sentence splits;Levenshtein similarity;Exact copies;Additions proportion;Deletions proportion;Lexical complexity score\n"
results=$"bleu;sari;sari_add;sari_keep;sari_del;fkgl;pbert_ref;rbert_ref;fbert_ref;pbert_src;rbert_src;fbert_src;ppl_mean;ppl_std;lens;lens_std;intra_dist1;intra_dist2;inter_dist1;inter_dist2;Compression ratio;Sentence splits;Levenshtein similarity;Exact copies;Additions proportion;Deletions proportion;Lexical complexity score;file_id\n"

gpu=${1:-"0"}
split=${2:-"test"}

export CUDA_VISIBLE_DEVICES=$gpu # recommended if computing ppl with gpt2

# # # re-evaluate ablation experiment outputs
# # for n in 10 50 100 500 1000 5000 10000 20000 30000 36989; do
# #     infile="resources/fudge/discriminators/newsela-lp_abl_${n}_l4_article_paragraphs/bart_large_muss_mined_en/newsela_abl_${n}_l4_article_paragraphs/newsela_manual_v0_v4_test/lambda1.0_pretopk200_beams5_estopFalse_maxl128_minl10_sampleFalse_lp1.0_norep1_bgrps1_nbest5_repp1.2_softFalse_temp1.0_topk0_topp1.0_bs1.txt"
# #     result_file="resources/results/newsela_abl_${n}_l4_article_paragraphs.csv"
# #     if [[ -f ${infile} ]]; then
# #         echo "scoring ${infile} ..."
# #         python evaluation/simplification_evaluation_v2.py ${infile} --src_file $src_files/newsela_manual_v0_v4_test.tsv --use_cuda --lens_model_path "/srv/scratch1/kew/llm_ats/LENS/checkpoints/epoch=5-step=6102.ckpt" --out_file $result_file
# #     else
# #         echo "file not found: ${infile}"
# #     fi
# # done

mkdir -p $SCRATCH/results

# ground truth evaluations
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
    # python evaluation/simplification_evaluation_v2.py ${tmpfile} --src_file $infile --out_file $outfile
    # remove tmpfile
    rm $tmpfile
done


# # # evaluate muss outputs on Newsela
tgt_dir=$muss_outputs
echo ""
echo "***** Current target dir: $tgt_dir *****"
echo ""
infiles=$(find $tgt_dir -name "newsela_manual_v0_v*_${split}_*.pred")
for infile in $infiles; do
    # extract level id from filename
    level=$(echo $infile | grep -oP '(?<=v0_v)[0-9]+')
    
    echo "scoring ${infile} with level ${level} ..."
    res=$(python evaluation/simplification_evaluation_v2.py ${infile} --src_file $src_files/newsela_manual_v0_v${level}_${split}.tsv --use_cuda --lens_model_path "/srv/scratch1/kew/llm_ats/LENS/checkpoints/epoch=5-step=6102.ckpt")
    [[ ! -z "$res" ]] && results+=$"${res:301}\n" || echo "failed to score ${infile}"
done

# supervised outputs
for train_data in "newsela_manual" "newsela_auto"; do
    tgt_dir="${super_outputs}/${train_data}/outputs/*${split}"
    echo ""
    echo "***** Current target dir: $tgt_dir *****"
    echo ""

    # get all relevant files in target output dir (should be only one, but eh)
    infiles=$(find $tgt_dir -name "lambda*.txt")
    for infile in $infiles; do
        # if the relevant file exists, run eval
        level=$(echo $infile | grep -oP '(?<=v0_v)[0-9]+')

        echo "scoring ${infile} with level ${level} ..."
        res=$(python evaluation/simplification_evaluation_v2.py ${infile} --src_file $src_files/newsela_manual_v0_v${level}_${split}.tsv --use_cuda --lens_model_path "/srv/scratch1/kew/llm_ats/LENS/checkpoints/epoch=5-step=6102.ckpt")
        [[ ! -z "$res" ]] && results+=$"${res:301}\n" || echo "failed to score ${infile}"
    done
done

# evaluate fudge outputs on Newsela
for disc_type in newsela-lp; do
    tgt_dir="${fudge_outputs}/${disc_type}_l*_article_paragraphs/*${split}"
    echo ""
    echo "***** Current target dir: $tgt_dir *****"
    echo ""
    # get all relevant files in target output dir
    infiles=$(find $tgt_dir -name "lambda*.txt")
    for infile in $infiles; do
        # extract level id from filename
        level=$(echo $infile | grep -oP '(?<=v0_v)[0-9]+')

        echo "scoring ${infile} with level ${level} ..."
        res=$(python evaluation/simplification_evaluation_v2.py ${infile} --src_file $src_files/newsela_manual_v0_v${level}_${split}.tsv --use_cuda --lens_model_path "/srv/scratch1/kew/llm_ats/LENS/checkpoints/epoch=5-step=6102.ckpt")
        [[ ! -z "$res" ]] && results+=$"${res:301}\n" || echo "failed to score ${infile}"
    done
done


# evaluate fsllm outputs on Newsela
for model in "TinyLlama_1.1B_intermediate_step_1431k_3T" "Llama_2_7b_hf"; do
    tgt_dir="$fsllm_outputs/$model"
    echo ""
    echo "***** Current target dir: $tgt_dir *****"
    echo ""
    
    # get all relevant files in target output dir
    infiles=$(find $tgt_dir -name "newsela_manual_v0_v*${split}-*.jsonl")
    for infile in $infiles; do
        # extract level id from filename
        level=$(echo $infile | grep -oP '(?<=v0_v)[0-9]+')

        echo "scoring ${infile} with level ${level} ..."
        res=$(python evaluation/simplification_evaluation_v2.py ${infile} --use_cuda --lens_model_path "/srv/scratch1/kew/llm_ats/LENS/checkpoints/epoch=5-step=6102.ckpt")
        # if the resul is not empyt, add to full results
        [[ ! -z "$res" ]] && results+=$"${res:301}\n" || echo "failed to score ${infile}"
    done
done


echo ""
echo "***** RESULTS *****"
echo -e $results | tee "resources/results/newsela_manual_${split}.csv"

echo "done"