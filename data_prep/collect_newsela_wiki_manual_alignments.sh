#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# This script extracts aligned sentences from the Newsela and Wiki corpora (wiki not yet tested)

# Usage:
# bash data_prep/collect_newsela_wiki_manual_alignments.sh [alignment_unit]

# Updated: 2023-04-06
# - support extracting aligned units other than sentences (e.g. paragraphs, docs)
# - support extracting alignments for all levels (grade 1-12)


SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
BASE=$SCRIPT_DIR/../

cd $BASE && echo "Current directory: $PWD"

alignment_unit=${1:-sent}

# article versions
echo "Extracting aligned ${alignment_unit}s for all versions..."
for split in train test dev; do
    echo "$split"
    for version in 1 2 3 4; do
        echo "0 - $version"
        python data_prep/extract_alignments_wiki_newsela_manual.py \
        --infile resources/data/en/newsela-auto/newsela-manual/all/$split.tsv \
        --corpus_dir resources/data/en/newsela_article_corpus_2016-01-29/ \
        --output_dir resources/data/en/aligned/ \
        --complex_level 0 --simple_level $version \
        --unit $alignment_unit
    done
done

# reading grade levels
echo "Extracting aligned ${alignment_unit}s for all levels..."
for split in train test dev; do
    echo "${split}"
    for complex_grade in 12 11 10 9 8 7 6 5 4 3 2; do
        for simple_grade in 1 2 3 4 5 6 7 8 9 10 11; do
            echo "${complex_grade} - ${simple_grade}"
            python data_prep/extract_alignments_wiki_newsela_manual.py \
                --infile resources/data/en/newsela-auto/newsela-manual/all/$split.tsv \
                --output_dir resources/data/en/aligned/ \
                --corpus_dir resources/data/en/newsela_article_corpus_2016-01-29/ \
                --complex_level ${complex_grade} --simple_level ${simple_grade} \
                --unit $alignment_unit \
                --grade_level
        done
    done
done


# for split in train test dev; do
#     # wiki only has 1 level
#     python extract_aligned_sents_wiki_newsela_manual.py \
#         --infile resources/data/en/wiki-auto/wiki-manual/${split}.tsv \
#         --outfile resources/data/en/aligned/wiki_manual_${split}.tsv \
#         --complex_level 0 --simple_level 1 --wiki
    
#     # newsela has 4 levels (versions)
#     for version in 1 2 3 4; do
#         python extract_aligned_sents_wiki_newsela_manual.py \
#             --infile resources/data/en/newsela-auto/newsela-manual/all/${split}.tsv \
#             --outfile resources/data/en/aligned/newsela-manual_sents/newsela_manual_v0_v${version}_${version}.tsv \
#             --complex_level 0 --simple_level $version
#     done
