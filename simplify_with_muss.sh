#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# Server: rattle

# bash simplify_with_muss.sh --src_file /srv/scratch1/kew/ats/data/en/aligned/newsela_manual_v0_v4_dev.tsv --out_path /srv/scratch1/kew/ats/muss/outputs --gpu 3
# python scripts/simplify.py scripts/examples.en --model-name muss_en_wikilarge_mined



CONDA_INIT=/home/user/kew/anaconda3/etc/profile.d/conda.sh
# expects muss to be installed in the `installs`` directory in simple_fudge/
MUSS_DIR=installs/muss


src_file=""
params_file=""
out_path=""
gpu="3"
len_ratio="0.75" # default
lev_sim="0.65" # default
word_rank="0.75" # default
tree_depth="0.4" # default

while [[ $# -gt 0 ]]; do
  case $1 in
    -s|--src_file)
      src_file="$2"
      shift # past argument
      shift # past value
      ;;
    -o|--out_path)
      out_path="$2"
      shift # past argument
      shift # past value
      ;;
    -p|--params)
      params_file="$2"
      shift # past argument
      shift # past value
      ;;
    --gpu)
      gpu="$2"
      shift # past argument
      shift # past value
      ;;
    --len_ratio)
      len_ratio="$2"
      shift # past argument
      shift # past value
      ;;
    --lev_sim)
      lev_sim="$2"
      shift # past argument
      shift # past value
      ;;
    --word_rank)
      word_rank="$2"
      shift # past argument
      shift # past value
      ;;
    --tree_depth)
      tree_depth="$2"
      shift # past argument
      shift # past value
      ;;
    -*|--)
      echo "Unknown option $1"
      exit 1
      ;;
    *)
      POSITIONAL_ARGS+=("$1") # save positional arg
      shift # past argument
      ;;
  esac
done

if [ -z "$src_file" ]
  then 
    echo "No source file provided for translating!" && exit 1
fi

# get stem of source file for out_file
src_file_stem=$(basename -- "$src_file")
src_file_stem="${src_file_stem%.*}"
echo $src_file_stem

out_file="$out_path/${src_file_stem}_lr${len_ratio}_ls${lev_sim}_wr${word_rank}_td${tree_depth}.txt"
echo $out_file

export CUDA_VISIBLE_DEVICES=$gpu

source $CONDA_INIT

conda activate muss

echo "Activated environment: $CONDA_DEFAULT_ENV ..."

# create temp file for cutting the source column from datasets 
tmpfile=$(mktemp /tmp/muss_src_file.XXXXXX)

# trim first column from file
cut -f 1 $src_file >| $tmpfile

echo "Translating $tmpfile ..."
echo "Writing to $out_file ..."

python $MUSS_DIR/scripts/simplify.py \
  $tmpfile \
  --model-name muss_en_mined \
  --out_file $out_file \
  --len_ratio $len_ratio \
  --lev_sim $lev_sim \
  --word_rank $word_rank \
  --tree_depth $tree_depth

# clean up tmpfile
rm $tmpfile

conda deactivate
echo "Finished simplifying with MUSS $src_file"
echo "Simplifications: $out_file"