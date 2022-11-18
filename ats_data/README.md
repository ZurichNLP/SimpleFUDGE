# Data handling scripts for Simple FUDGE and English ATS.

### Training Data

To prepare data for training FUDGE discriminators for simplification

**WIKI data:** (best to run this on a large server)

```
bash prepare_wiki_data_for_fudge.sh
```

**Newsela:**

```
bash prepare_newsela_data_for_fudge.sh
```

### Evaluation data

#### Newsela Corpus (Xu et al. 2015)

News articles manually simplified to up to 5 levels of simplicity
Access must be requested at https://newsela.com/data

For the sentence alignments, we use the high-quality,
manually aligned splits from Jiang et al. (2020).

```
bash collect_newsela_wiki_manual_alignments.sh
```


#### Turk Corpus (Xu et al. 2016)

Corpus of (mainly) lexical paraphrasing from English
Wikipedia. It contains 2,359 original sentences, each with 8 manual reference simplifications.
Dataset consists of two subsets:
    - 2,000 for validation 
    - 359 for testing

```
git clone git@github.com:cocoxu/simplification.git ./data/turk
```

#### HSplit (Sulem et al., 2018)

Corpus of split/structurally simplified versions of 359 sentences
from the Turk Corpus test set.

```
git clone git@github.com:eliorsulem/HSplit-corpus.git ./data/hsplit
```

#### ASSET (Alva-Manchego et al., 2020)

Corpus of Multiple simplification transformations versions
of the Turk Corpus. See
https://huggingface.co/datasets/asset for full description.

```
git clone git@github.com:facebookresearch/asset.git ./data/asset
```

<!-- - TURK test (Xu et al. 2016) 
        - https://cocoxu.github.io/publications/tacl2016-smt-simplification.pdf)
        - https://huggingface.co/datasets/turk
    
    - Newsela (Xu et al. 2015)
        - https://aclanthology.org/Q15-1021.pdf
        - aligned sentences collected from
          newsela_articles_*.aligned.sents.txt and
          aggregated by tgt level

    - ASSET  -->