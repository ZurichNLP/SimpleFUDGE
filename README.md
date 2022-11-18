# SimpleFUDGE

This repo contains the source code for the paper "Target-Level Sentence Simplification as Controlled Paraphrasing", presented at the Workshop on Text Simplification, Accessibility, and Readability (TSAR-2022).

## Overview

The idea in the paper is to apply [Future Discriminators for Generation (FUDGE)](https://arxiv.org/abs/2104.05218) to steer paraphrastic generations towards simpler alternatives.

For this study, we reimplemented FUDGE as a standalone, [custom LogitProcessor](https://towardsdatascience.com/the-power-of-constrained-language-models-cf63b65a035d) which can be used with most of HuggingFace's generation utilities. This facilitates experimentation with more various decoding techniques, e.g. beam search, top-k/top-p sampling.

This repo is setup as follows:

- `analysis`: contain jupyter notebooks for analysing and plotting results.
- `ats_data`: scripts for preparing various simplification data sets.
- `easse_patch`: a single script which can be used to replace one of the original scripts in EASSE.
- `legacy`: original scripts and files from the [original FUDGE repo](https://github.com/yangkevin2/naacl-2021-fudge-controlled-generation)
- implementation and experimentation scripts.

## Setup

```
conda create -n fudge python=3.8.5 -y
pip install -r requirements.txt

git clone git@github.com:feralvam/easse.git
cd easse
pip install -e .
```

Note, to help us inspect quality estimation outputs, we provide a modified version of EASSE's `quality_estimation.py` which includes the option to not aggregate the computed QE metrics on the corpus level. 
To make use of this, replace the `quality_estimation.py` in the EASSE package with our version located in `easse_patch/`.

**Note:** Paths are typically hardcoded in the scripts. Before running, adjust these to suit your system.

## Running paper experiments

The script `run_experiments.sh` contains all experiment commands defined as functions with example calls on how to execute them.

The basic workflow consists of the following steps:

### Prepare Data

For our experiments, we train our FUDGE classifiers on the Newsela Corpus (Xu et al. 2015). Access must be requested [here](https://newsela.com/data/). However, any comparable labelled data can be used, e.g. Simple Endlish Wiki.

For the evaluating with aligned sentences, we use the manually aligned splits from Jiang et al. (2020).

```
cd ats_data

# prepare training data
bash prepare_newsela_data_for_fudge.sh

# prepare evaluation (sentence-aligned) data
bash collect_newsela_wiki_manual_alignments.sh
```

### FUDGE Discriminator Model Training

FUDGE model training is defined in `main.py`. To train a FUDGE discriminator on paragraph subsequences, i.e. including multi-sentence subsequences, run

```
# newsela discriminator on paragraph-level with level 4 simplifications
nohup bash run_experiments.sh train_simple_newsela_discriminator 2 4 article_paragraphs >| logs/train_l4.log &
```

### Generator Model Fine-Tuning

To train a paraphrastic generator, we fine-tune BART with web-mined paraphrases from [Martin et al., 2020](https://arxiv.org/abs/2005.00352). Big thanks to Louis Martin for helping us get this data!

```
# bart-large paraphraser trained on muss mined en
nohup bash run_experiments.sh finetune_bart_large_on_muss_mined >| logs/finetune.log &
```

### Model Inference

Once we have the generator model and a FUDGE discriminator, we can get simplifications using `predict_simplify.py`, e.g.:

```
python predict_simplify.py \
    --condition_model <PATH_TO_FUDGE_DISCRIMINATOR> \
    --generation_model <PATH_TO_GENERATION_MODEL> \
    --condition_lambda 5 \
    --num_beams 1 --num_return_sequences 1 \
    --input_text "Memorial West's class is one of several programs offered through hospitals to help children stay healthy through exercise and proper eating"
```

To decode a test set, run:

```
python inference.py \
    --condition_model <PATH_TO_FUDGE_DISCRIMINATOR> \
    --generation_model <PATH_TO_GENERATION_MODEL> \
    --infile <PATH_TO_INFILE> \
    --batch_size 10 --condition_lambda 5
```

**NOTE:** We assume the test set to be a `.txt` or `.tsv` file. If a `.tsv` file is passed, it's assumed that the first column contains the complex sentences to be simplified, e.g.:

```
VIRGINIA CITY, Nev. — One wonders what Mark Twain....     VIRGINIA CITY, Nev. — Mark Twain is a famous American writer...
```

### Evaluating Outputs

For evaluating generated simplifications, we use [EASSE](https://github.com/feralvam/easse)

The relevant metrics are computed in `simplification_evaluation.py`. To run evaluations, run:

```
python simplification_evaluation.py \
    --src_file <PATH_TO_TEST_SET> \
    --hyp_file <PATH_TO_MODEL_OUTPUTS>
```

## Additional Info

### Hyperparameter Search

FUDGE has one main hyperparameter (lambda, default=1). Selecting the correct value for lamba may depend on the quality of the paraphraser, discriminator and the corpus. 

The script `hp_search.py` can be used to perform a hyperparameter sweep. We also provide the notebook `analyse_hp_sweeps.ipynb` to inspect and plot the results.

<!-- ## Sanity check

Output of new implementation matches the original when using
greedy decoding:

```
# note, these commands below are now deprecated
python predict_simplify_as_logits_processor.py --ckpt /srv/scratch6/kew/fudge/ckpt/simplify/simplify_l4_v3/model_best.pth.tar --dataset_info /srv/scratch6/kew/fudge/ckpt/simplify/simplify_l4_v3/dataset_info --precondition_topk 200 --condition_lambda 80 --vectorized --num_beams 1 --num_return_sequences 1 --input_text "This is a test sentence"
['They are saying.This is a test.']

python predict_simplify.py --ckpt /srv/scratch6/kew/fudge/ckpt/simplify/simplify_l4_v3/model_best.pth.tar --dataset_info /srv/scratch6/kew/fudge/ckpt/simplify/simplify_l4_v3/dataset_info --precondition_topk 200 --condition_lambda 80 --input_text "This is a test sentence"
['They are saying.This is a test.</s>']
``` -->

## Comparisons with Baselines

If you have [MUSS](https://github.com/facebookresearch/muss) installed and running (e.g. in a separate conda environment), you can adapt `simplify_with_muss.sh` to generate simplifications for a given input file, e.g.:

```
bash simplify_with_muss.sh /srv/scratch6/kew/ats/data/en/aligned/turk_test.tsv /srv/scratch6/kew/ats/muss/outputs/turk_test_HEAD.txt 5
```

To train the label-supervised method, run:

```
nohup bash run_experiments.sh finetune_bart_large_on_supervised_labeled_newsela_manual >| newsela_supervised_finetune.log &
```

## Citation

```

```
