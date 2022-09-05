# SIMPLE FUDGE

This repo adapts the central idea behind FUDGE: Controlled
Text Generation With Future Discriminators
(https://arxiv.org/abs/2104.05218) by Kevin Yang and Dan
Klein (2021) for text simplification.

# Changes

FUDGE has been reimplemented here as a standalone 
LogitProcessor which can be used with most of
HuggingFace's generation utilities. This facilitates
experimentation with more advanced decoding techniques, e.g.
beam search, top-k/nucleus sampling.


## Relevant links

- [Original code](https://github.com/yangkevin2/naacl-2021-fudge-controlled-generation)
- [Custom logits processors](https://towardsdatascience.com/the-power-of-constrained-language-models-cf63b65a035d)

## Setup

```
conda create -n fudge python=3.8.5 -y
pip install -r requirements.txt

git clone git@github.com:feralvam/easse.git
cd easse
pip install -e .
```

Note, to better inspect quality estimation outputs, we provide a modified version of EASSE's `quality_estimation.py` which includes the option to not aggregate the computed QE metrics on the corpus level. 
To make use of this, replace the default `quality_estimation.py` with the version located in `easse_patch/`.

## Data 

For information on the simplification data used, see the relevant README in [`ats_data`](https://github.com/tannonk/fudge/blob/master/ats_data/README.md)

## Usage Examples

```
GENERATION_MODEL=/srv/scratch6/kew/ats/fudge/generators/bart_large_paraNMT_filt_fr 
CONDITION_MODEL=/srv/scratch6/kew/ats/fudge/discriminators/wiki100M_bart_glove/model_best.pth.tar
# OR
# CONDITION_MODEL=/srv/scratch6/kew/ats/fudge/discriminators/newsela4_bart_glove/model_best.pth.tar

python predict_simplify.py \
    --condition_model $CONDITION_MODEL \
    --generation_model $GENERATION_MODEL \
    --precondition_topk 200 \
    --condition_lambda 10 \
    --vectorized \
    --num_beams 4 \
    --num_return_sequences 4 \
    --soft

```

## HP Search

FUDGE has one main hyperparameter (lambda, default=1). Selecting the
correct value for lamba may depend on the quality of the
paraphraser, discriminator and the corpus. 

See the script `hp_search.py` for a hyperparameter sweep and
the analysis in the notebook.

## Sanity check

Output of new implementation matches the original when using
greedy decoding:

```
# note, these commands below are now deprecated
python predict_simplify_as_logits_processor.py --ckpt /srv/scratch6/kew/fudge/ckpt/simplify/simplify_l4_v3/model_best.pth.tar --dataset_info /srv/scratch6/kew/fudge/ckpt/simplify/simplify_l4_v3/dataset_info --precondition_topk 200 --condition_lambda 80 --vectorized --num_beams 1 --num_return_sequences 1 --input_text "This is a test sentence"
['They are saying.This is a test.']

python predict_simplify.py --ckpt /srv/scratch6/kew/fudge/ckpt/simplify/simplify_l4_v3/model_best.pth.tar --dataset_info /srv/scratch6/kew/fudge/ckpt/simplify/simplify_l4_v3/dataset_info --precondition_topk 200 --condition_lambda 80 --input_text "This is a test sentence"
['They are saying.This is a test.</s>']
```

## Comparisons with other Simplification Methods

if [MUSS](https://github.com/facebookresearch/muss) is installed and running in a separate conda
environment, use `simplify_with_muss.sh` to generate
simplifications for a given input file, e.g.

```
bash simplify_with_muss.sh /srv/scratch6/kew/ats/data/en/aligned/turk_test.tsv /srv/scratch6/kew/ats/muss/outputs/turk_test_HEAD.txt 5
```

## Evaluating Outputs

For evaluating generated simplifications, we use [EASSE](https://github.com/feralvam/easse)

The relevant metrics are within `simplification_evaluation.py`, e.g.

```
python simplification_evaluation.py \
    --src_file /srv/scratch6/kew/ats/data/en/aligned/turk_test.tsv \
    --hyp_file /srv/scratch6/kew/ats/fudge/results/bart_large_paraNMT_filt_fr/turk_test/lambda0.0_pretopk200_beams4_estopFalse_maxl128_minl1_sampleFalse_lp1.0_norep1_bgrps1_nbest1_repp1.0_softFalse_temp1.0_topk0_topp1.0.txt
```

## Model Training

### Discriminator Model Training

```
# newsela discriminator on sentence-level with level 4 simplifications
nohup bash run_experiments.sh train_simple_newsela_discriminator 2 4 article_sents >| train_l4.log &

# newsela discriminator on paragraph-level with level 4 simplifications
nohup bash run_experiments.sh train_simple_newsela_discriminator 2 4 article_paragraphs >| train_l4.log &
```

### Generator Model Fine-Tuning

```
# bart-large paraphraser trained on muss mined en
nohup bash run_experiments.sh finetune_bart_large_on_muss_mined >| finetune.log &
```


