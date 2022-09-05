#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# Wraps call to simplification evaluation for all relevant outputs in one script, e.g.
# python simplification_evaluation.py \
#     --src_file /srv/scratch6/kew/ats/data/en/aligned/newsela_manual_v0_v1_dev.tsv \
#     --hyp_file /srv/scratch6/kew/ats/muss/outputs/newsela_manual_v0_v1_dev*.pred


src_files=/srv/scratch6/kew/ats/data/en/aligned
muss_outputs=/srv/scratch6/kew/ats/muss/outputs
super_outputs=/srv/scratch6/kew/ats/supervised
fudge_outputs=/srv/scratch6/kew/ats/fudge/results/bart_large_muss_mined_en
# outfile=${1:"results.csv"}

# init results as header from evaldataframe
# results=$"file;ppl_diff;bleu;sari;fkgl;bertscore_p;bertscore_r;bertscore_f1;Compression ratio;Sentence splits;Levenshtein similarity;Exact copies;Additions proportion;Deletions proportion;Lexical complexity score\n"
results=$"file;ppl;bleu;sari;fkgl;bertscore_p_ref;bertscore_r_ref;bertscore_f1_ref;bertscore_p_src;bertscore_r_src;bertscore_f1_src;intra_dist1;intra_dist2;inter_dist1;inter_dist2;Compression ratio;Sentence splits;Levenshtein similarity;Exact copies;Additions proportion;Deletions proportion;Lexical complexity score\n"

split=${1:-"test"}

export CUDA_VISIBLE_DEVICES="6" # recommended if computing ppl with gpt2

# evaluate muss outputs on Newsela
for level in 1 2 3 4; do
    # for split in "dev" "test"; do
        echo "scoring $muss_outputs/newsela_manual_v0_v${level}_${split}*.pred"
        res=$(python simplification_evaluation.py --src_file $src_files/newsela_manual_v0_v${level}_${split}.tsv --hyp_file $muss_outputs/newsela_manual_v0_v${level}_${split}*.pred  --compute_ppl)
        results+=$"${res:307}\n" # slice string to remove header
        echo -e $results
    # done
done

# supervised outputs
for train_data in "newsela_manual" "newsela_auto"; do
    for level in 1 2 3 4; do
        # for split in "dev" "test"; do
        tgt_dir="$super_outputs/$train_data/results/newsela_manual_v0_v${level}_${split}"
        echo "current target dir: $tgt_dir"
        if [[ -d $tgt_dir ]]; then
            # get all relevant files in target output dir
            hyp_files=$(find $tgt_dir -name "lambda*.txt")
            for hyp_file in $hyp_files; do
                echo "scoring $hyp_file ..."
                # if the relevant file exists, run eval
                if [[ -f "$hyp_file" ]]; then 
                    res=$(python simplification_evaluation.py --src_file $src_files/newsela_manual_v0_v${level}_${split}.tsv --hyp_file $hyp_file --compute_ppl)
                    # if the resul is not empyt, add to full results
                    [[ ! -z "$res" ]] && results+=$"${res:307}\n" || echo "failed to score $hyp_file"
                fi
            done
        fi
        # done
    done
done

# evaluate fudge outputs on Newsela
for disc_type in "article_para_sents" "article_paragraphs"; do
    for fudge_format in "newsela" "newsela-lp"; do # comparison between training on lineparts and full lines
        for cond_level in 1 2 3 4; do
            for split_level in 1 2 3 4; do # we decode all available split levels with conditioners trained on all available levels
                # for split in "dev" "test"; do
                tgt_dir="$fudge_outputs/${fudge_format}_l${cond_level}_${disc_type}/newsela_manual_v0_v${split_level}_${split}"
                if [[ -d $tgt_dir ]]; then
                    echo "current target dir: $tgt_dir"
                    # get all relevant files in target output dir
                    hyp_files=$(find $tgt_dir -name "lambda*.txt")
                    for hyp_file in $hyp_files; do
                        echo "scoring $hyp_file ..."
                        # if the relevant file exists, run eval
                        if [[ -f "$hyp_file" ]]; then 
                            res=$(python simplification_evaluation.py --src_file $src_files/newsela_manual_v0_v${split_level}_${split}.tsv --hyp_file $hyp_file --compute_ppl)
                            # if the resul is not empyt, add to full results
                            [[ ! -z "$res" ]] && results+=$"${res:307}\n" || echo "failed to score $hyp_file"
                            # echo -e $results
                        fi
                    done
                fi
                # done
            done
        done
    done
done


echo ""
echo "***** RESULTS *****"
# echo -e $results
echo -e $results | tee "results/${split}.csv"


