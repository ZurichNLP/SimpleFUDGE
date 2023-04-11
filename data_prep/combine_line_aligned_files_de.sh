#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# merge line-aligned files into single tsv files for compatibility with en data format

data_dir="/srv/scratch1/kew/ats/data/de/aligned/apa_capito"

for level in a1 a2 b1; do
    for split in train test dev; do
        paste $data_dir/or-$level/$split.or-$level.or $data_dir/or-$level/$split.or-$level.$level >| $data_dir/../apa_capito_${level}_$split.tsv
    done
done

echo "done: $data_dir"
