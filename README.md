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

## Experiments

- [ ] improved classifier
    - [x] initialised from pre-trained glove embeddings
    - [x] increasing model size
    - [x] bidirectional 
    - [ ] using pretrained model, e.g. BERT / BART
- [ ] improving paraphraser
    - [ ] training on mined paraphrase sentences à la [MUSS](https://github.com/facebookresearch/muss)

## TODOs

- [ ] experiment with improved classifier (e.g.
  BERT/BART-base) 
- [ ]

<!-- ## TODOs
- [ ] fix model string inherited from dataset for BART model
- [ ]  -->

<!-- ## Setup/Installation

We tested on Python 3.8.5 but earlier versions of Python 3 are almost certainly fine. To get the required packages (other versions likely to work too):

```
pip install -r requirements.txt
```

Additionally, to get our pre-trained predictor checkpoints and training data, run:

```
wget https://naacl2021-fudge-files.s3.amazonaws.com/large_files.zip
```

and extract the zip to the top-level `lm-prediction/` folder. (There should be three folders, `ckpt/`, `train_data/`, and `topic_human_evals/`. The zip is 7GB.)

`ckpt/` contains predictor checkpoints for each task if you are just interested in running inference. (Note that for the paper results, we used predictors trained with an older version of the code, but the new checkpoints get similar results, so you are OK to use the new predictors provided here if e.g. you just want to use FUDGE as a baseline. You can just run the evaluation commands provided below; it should take maybe 5-60 minutes depending on the task and your compute, assuming you have a GPU.)

`train_data/` contains our GPT2-generated training data for the poetry and topic tasks' predictors. See https://github.com/raosudha89/GYAFC-corpus for instructions on gaining access to the GYAFC data used for the machine translation formality task; replace our dummy folders with the corresponding folders/files if you want to train our formality predictor. 
 -->
<!-- ## Poetry Couplet Completion -->

<!-- ### Evaluation

To generate outputs, run:

```
python -u evaluate_poetry.py --iambic_ckpt ckpt/poetry/iambic_predictor/model.pth.tar --rhyme_ckpt ckpt/poetry/rhyme_predictor/model.pth.tar --newline_ckpt ckpt/poetry/newline_predictor/model.pth.tar --dataset_info ckpt/poetry/rhyme_predictor/dataset_info --rhyme_info ckpt/poetry/rhyme_predictor/rhyme_info --prefix_file poetry_data/couplet_prefixes.txt --precondition_topk 200 > poetry_preds.log
```

Then evaluate metrics using:

```
python eval_poetry_metrics.py --pred_file poetry_preds.log --prefix_file poetry_data/couplet_prefixes.txt
```

### Training your own predictors

Example commands for all three predictors used in the poetry task below. (You actually probably don't need so many epochs for iambic and rhyme; in any case the commands will save intermediate ckpts so you can just stop them early if needed by inspecting the log.)

Iambic predictor:

```
python -u main.py --task iambic --data_dir train_data/gpt2_generations --save_dir ckpt/poetry/iambic_retrain_predictor --num_workers 20 --batch_size 128 --epoch_max_len 100000 --validation_freq 10  --lr 2e-4 --epochs 1500 > iambic_retrain_predictor.log
```

Rhyme predictor:

```
python -u main.py --task rhyme --data_dir train_data/gpt2_generations --save_dir ckpt/poetry/rhyme_retrain_predictor --num_workers 20 --batch_size 128 --epoch_max_len 100000 --validation_freq 10  --lr 2e-4 --epochs 1500 > rhyme_retrain_predictor.log
```

End of sentence predictor (referred to as "newline" in the code; 50 epochs is more than enough for this one):

```
python -u main.py --task newline --data_dir train_data/gpt2_generations --save_dir ckpt/poetry/newline_retrain_predictor --num_workers 20 --batch_size 128 --epoch_max_len 100000 --validation_freq 10  --lr 2e-4 --epochs 50 > newline_retrain_predictor.log
```

The same evaluation commands as before will work; just modify the paths in the command to point to `model_best.pth.tar`, `dataset_info`, and `rhyme_info` from your newly trained ckpt folders. 

## Topic Control

### Evaluation

To generate outputs, run:

```
python -u evaluate_topic.py --ckpt ckpt/topic/future_word_predictor/model.pth.tar --dataset_info ckpt/topic/future_word_predictor/dataset_info --prefix_file topic_data/topic_prefixes.txt --wordlist_dir topic_data/wordlists --condition_lambda 4.0 --verbose --precondition_topk 200 --topk 10 --sample_size 3 --max_sample_batch 1 --length_cutoff 80 --log_file topic_preds.log
```

Then evaluate metrics using:

```
python eval_topic_metrics.py --log_file topic_preds.log --tw_dir topic_data/test_wordlists
```

You can also find our original generations and baselines in `topic_human_evals/`.

### Training your own predictors

Example command below.

```
python -u main.py --task topic --data_dir train_data/gpt2_generations --save_dir ckpt/topic/future_word_retrain_predictor --num_workers 20 --batch_size 128 --epoch_max_len 100000 --validation_freq 10  --lr 2e-4 --epochs 500 --glove_file train_data/glove.840B.300d.txt > future_word_retrain_predictor.log
```

The same evaluation commands as before will work; just modify the paths in the command to point to `model_best.pth.tar`, `dataset_info`, and `rhyme_info` from your newly trained ckpt folders. 

## Machine Translation Formality

### Evaluation

To generate outputs, run:

```
python -u evaluate_formality.py --ckpt ckpt/formality/predictor_gyafc_entertainment_music/model.pth.tar --dataset_info ckpt/formality/predictor_gyafc_entertainment_music/dataset_info --in_file formality_data/fisher_test_oracle.es --model_path ckpt/formality/marian_finetune_fisher > formality_preds.log
```

The above command generates predictions using the Marian model finetuned on the Fisher dataset; remove the `--model_path` argument to get predictions with the un-finetuned Marian model from HuggingFace (referred to as 0-shot in the paper)

Then evaluate metrics using:

```
python eval_formality_metrics.py --pred formality_preds.log --ref formality_data/test.noid.cleaned_0 formality_data/test.noid.cleaned_1 --ckpt ckpt/formality/test_evaluator_gyafc_family_relationships/model.pth.tar --dataset_info ckpt/formality/test_evaluator_gyafc_family_relationships/dataset_info
```

### Training your own predictors

Example command below. (Reminder: you need to go get the GYAFC dataset following the instructions in https://github.com/raosudha89/GYAFC-corpus.)

```
python -u main.py --task formality --data_dir train_data/GYAFC_Corpus/Entertainment_Music --save_dir ckpt/formality/formality_retrain_predictor --num_workers 20 --batch_size 32 --epoch_max_len 1000000 --validation_freq 1 --lr 2e-5 --epochs 20 > formality_retrain_predictor.log
```

(The test-time formality evaluator is trained in the same way, just using the Family/Relationships half of the GYAFC dataset.)

The same evaluation commands as before will work; just modify the paths in the command to point to `model_best.pth.tar`, `dataset_info`, and `rhyme_info` from your newly trained ckpt folders. 

## Running FUDGE on your own data

The code has been refactored so that the iambic (poetry), rhyme (poetry), newline (poetry), future word (topic), and formality (machine translation) are controlled by the `--task` flag to `main.py`. You should add your task as another option here, then modify the data processing in `data.py` and the model in `model.py` as needed for your task. (In `data.py` you probably won't need all the entries of the tuple that is expected of the loader; you can just put dummy entries in the ones you don't need.) You might also need to modify the loss computation in the `train` and `validate` functions in `main.py`. You'll probably want to write new evaluation scripts, though the existing poetry/topic/formality ones are hopefully helpful as references. 

Alternatively, the general FUDGE framework is pretty simple, so you could always try reimplementing things yourself. A few additional details based on questions I've received: 

(1) The formality task setup is likely closest to what you want if you're just trying to run the simplest form of FUDGE (take a language model, and use a classifier to optimize toward a single attribute) although you may need to swap out the Marian translation model/tokenizer we use. 

(2) When you construct your training data, if you have an example in your data e.g. "This movie is great!" for positive sentiment, you want to learn on all the pairs (This, +), (This movie, +), (This movie is, +), etc., as that's one of the main points of our approach. 

(3) For computational efficiency, we first filter the base model's next token probabilities down to the top 200 (Sec. 3.1 in the paper), before adding the classifier logits. This way you only need to evaluate your classifier on 200 continuations. Then afterward, you filter down again to whatever top-k/greedy/nucleus sampling you're using for evaluation (we use top-k with k=10 for poetry and topic, greedy for formality). 

(4) You can use a pretrained LM backbone instead of a simple LSTM backbone for the predictor as well. This should work better when your dataset is smaller.

--- -->

