#!/usr/bin/env bash
# -*- coding: utf-8 -*-

for split in train test dev; do
    # wiki only has 1 level
    python extract_aligned_sents_wiki_newsela_manual.py \
        --infile /srv/scratch6/kew/ats/data/en/wiki-auto/wiki-manual/${split}.tsv \
        --outfile /srv/scratch6/kew/ats/data/en/aligned/wiki_manual_${split}.tsv \
        --complex_level 0 --simple_level 1 --wiki
    # newsela has 4 levels
    for level in 1 2 3 4; do
        python extract_aligned_sents_wiki_newsela_manual.py \
            --infile /srv/scratch6/kew/ats/data/en/newsela-auto/newsela-manual/all/${split}.tsv \
            --outfile /srv/scratch6/kew/ats/data/en/aligned/newsela_manual_v0_v${level}_${split}.tsv \
            --complex_level 0 --simple_level $level
    done
done

echo "done!"