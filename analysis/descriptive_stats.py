#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
from typing import List
import spacy
import textdescriptives as td
import pandas as pd

# load your favourite spacy model (remember to install it first using e.g. `python -m spacy download en_core_web_md`)
nlp = spacy.load("en_core_web_md")
nlp.add_pipe("textdescriptives/all")

def get_docs(texts: List[str], nlp=nlp, n_process: int = 4) -> List[spacy.tokens.Doc]:
    """
    Process a list of texts using spacy.
    """
    docs = nlp.pipe(texts, n_process=n_process)
    return docs

def get_descriptive_stats(docs: List[spacy.tokens.Doc]) -> pd.DataFrame:
    """
    Extract all metrics
    """
    df = td.extract_df(docs)
    return df

def clean_texts(texts: List[str]) -> List[str]:
    """
    Remove html paragrah tags and extra whitespace.
    """
    clean_texts = []
    for text in texts:
        text = re.sub(r"<p>", "", text)
        text = re.sub(r"</p>", r"\n\n", text)
        clean_texts.append(text.strip())
    return clean_texts

def process_texts(texts: List[str], n_process: int = 8) -> pd.DataFrame:
    """
    Process a list of texts and return a dataframe with descriptive statistics.
    """
    texts = clean_texts(texts)
    df = get_descriptive_stats(get_docs(texts, n_process=n_process))
    return df

def convert_to_percentage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert all columns with values between 0 and 1 to percentages.
    """
    for col in df.columns:
        if df[col].max() < 1 and df[col].min() >= 0:
            print(col)
            df[col] = df[col] * 100
    return df

lexical = [
    'oov_ratio',
    'pos_prop_PROPN',
    'pos_prop_PUNCT',
    'pos_prop_SCONJ',
    'pos_prop_PRON',
    'pos_prop_AUX',
    'pos_prop_ADP',
    'pos_prop_VERB',
    'pos_prop_CCONJ',
    'pos_prop_NOUN',
    'pos_prop_ADV',
    'pos_prop_ADJ',
    'pos_prop_DET',
    'pos_prop_SPACE',
    'pos_prop_PART',
    'pos_prop_NUM',
    # 'pos_prop_INTJ',
    # 'pos_prop_SYM',
    # 'pos_prop_X',
]

syntactic = [
    'dependency_distance_mean', # measure of syntactics complexity, the greater the distance, the more complex
    # 'dependency_distance_std',
    'prop_adjacent_dependency_relation_mean',
    # 'prop_adjacent_dependency_relation_std',
]

# human readable column names
name_map = {
    'first_order_coherence': 'First order coherence',
    'second_order_coherence': 'Second order coherence',
    'entropy': 'Entropy',
    'perplexity': 'Perplexity',
    'per_word_perplexity': 'Perplexity (per word)',
    'pos_prop_PROPN': 'Proportion of PROPN',
    'pos_prop_PUNCT': 'Proportion of PUNCT',
    'pos_prop_SCONJ': 'Proportion of SCONJ',
    'pos_prop_PRON': 'Proportion of PRON',
    'pos_prop_AUX': 'Proportion of AUX',
    'pos_prop_ADP': 'Proportion of ADP',
    'pos_prop_VERB': 'Proportion of VERB',
    'pos_prop_CCONJ': 'Proportion of CCONJ',
    'pos_prop_NOUN': 'Proportion of NOUN',
    'pos_prop_ADV': 'Proportion of ADV',
    'pos_prop_ADJ': 'Proportion of ADJ',
    'pos_prop_DET': 'Proportion of DET',
    'pos_prop_SPACE': 'Proportion of SPACE',
    'pos_prop_PART': 'Proportion of PART',
    'pos_prop_NUM': 'Proportion of NUM',
    'token_length_mean': 'Avg. token length',
    'token_length_median': 'Median token length',
    'token_length_std': 'Std. token length',
    'sentence_length_mean': 'Avg. sentence length',
    'sentence_length_median': 'Median sentence length',
    'sentence_length_std': 'Std. sentence length',
    'syllables_per_token_mean': 'Avg. syllable per token',
    'syllables_per_token_median': 'Median syllable per token',
    'syllables_per_token_std': 'Std. syllables per token',
    'n_tokens': 'Number of tokens',
    'n_unique_tokens': 'Number of unique tokens',
    'proportion_unique_tokens': 'Proportion unique tokens',
    'n_characters': 'Number of characters',
    'n_sentences': 'Number of sentences',
    'passed_quality_check': 'Quality check',
    'n_stop_words': 'Number of stop words',
    'alpha_ratio': 'Alpha Ratio',
    'mean_word_length': 'Avg. word length',
    'doc_length': 'Document length',
    'symbol_to_word_ratio_#': 'Symbol to word ratio',
    'proportion_ellipsis': 'Proportion of ellipsis',
    'proportion_bullet_points': 'Proportion of bullet points',
    'contains_lorem ipsum': 'Contains lorem ipsum',
    'duplicate_line_chr_fraction': 'Duplicate lines character fraction',
    'duplicate_paragraph_chr_fraction': 'Duplicate paragraphs character fraction',
    'duplicate_ngram_chr_fraction_5': 'Duplicate 5-gram character fraction',
    'duplicate_ngram_chr_fraction_6': 'Duplicate 6-gram character fraction',
    'duplicate_ngram_chr_fraction_7': 'Duplicate 7-gram character fraction',
    'duplicate_ngram_chr_fraction_8': 'Duplicate 8-gram character fraction',
    'duplicate_ngram_chr_fraction_9': 'Duplicate 9-gram character fraction',
    'duplicate_ngram_chr_fraction_10': 'Duplicate 10-gram character fraction',
    'top_ngram_chr_fraction_2': 'Top 2-gram character fraction',
    'top_ngram_chr_fraction_3': 'Top 3-gram character fraction',
    'top_ngram_chr_fraction_4': 'Top 4-gram character fraction',
    'oov_ratio': 'OOV Ratio',
    'flesch_reading_ease': 'Flesch Reading Ease',
    'flesch_kincaid_grade': 'FKGL',
    'smog': 'SMOG',
    'gunning_fog': 'Gunning-Fog',
    'automated_readability_index': 'Automated readability index',
    'coleman_liau_index': 'Coleman-Liau index',
    'lix': 'Lix',
    'rix': 'Rix',
    'dependency_distance_mean': 'Avg. dependency distance',
    'dependency_distance_std': 'Std. dependency distance',
    'prop_adjacent_dependency_relation_mean': 'Avg. proportion of adjacent depence relations',
    'prop_adjacent_dependency_relation_std': 'Std. proportion of adjacent depence relations',
    'pos_prop_INTJ': 'Proportion of INTJ',
    'pos_prop_SYM': 'Proportion of SYM',
    'level': 'Newsela Level',
    'pos_prop_X': 'Proportion of X',
}

if __name__ == "__main__":

    texts = [
        """
        <p>HONOLULU, Hawaii — Many homeless people move from colder states to Hawaii where it is warm and they can sleep on the beach. David Ige is the governor of Hawaii. He says it has become a big problem. There are too many homeless people in Hawaii. Ige has declared an emergency. He says the state must help the homeless.</p><p>Ige's emergency declaration will help the state government to quickly build a homeless shelter for families.</p><p>The governor's announcement came just days after officials cleared one of the nation's largest homeless camps. As many as 300 people were living in tents. Workers took down the camp.</p><p>The workers helped 125 people find housing, including 25 families. Some people rode buses to homeless shelters. Others moved into longer-term houses, Ige said.</p><p>"They are definitely off the streets and in a better situation," he added.</p><p>## New Shelter And Special Program Will Help</p><p>Yet many people still need homes. There is especially not enough houses for families.</p><p>There are 7,260 homeless people living in Hawaii. It has the highest rate of homelessness of any state in the United States.</p><p>Scott Morishige is the state homelessness coordinator in Hawaii. The number of families without a home almost doubled in the past year, he said.</p><p>The state will spend $1.3 million to help homeless people, Morishige said. The money will pay for the new shelter. It will pay for another program called Housing First. The program provides homes and services to people who have been homeless more than once. The money will also help families pay rent.</p><p>## A Place To Stay For Now</p><p>Meanwhile, workers are putting up a new homeless shelter. This short-term shelter is located on Sand Island. The rooms have just enough space for two people.</p><p>Each room will be just as big or bigger than a tent.</p><p>The rooms were made from shipping containers. Each large rectangular container has a window and a screen door. People will be able to sit outside under a shady spot, said Russ Wozniak. He is an architect and an engineer with Group 70. His company helped designed the shelter. The units will not have air conditioning. However, a special coating on the outside will help the inside stay cool.</p><p>Nearby will be a trailer that holds five bathrooms. Each has a toilet and shower.</p><p>The shelter will be finished in December. It will house up to 87 people at a time until they find permanent homes.</p><p>
        """,
        """
        <p>HONOLULU, Hawaii — Many homeless people move to Hawaii. The islands are warm. Homeless people do not have a place to live. Many of them can sleep on the beach.</p><p>David Ige is the governor of Hawaii. He is a leader in the state. He said there are too many homeless people. David said the government must help the homeless.</p><p>There were 300 people living in a camp. They were told to leave. Workers took down the camp. The workers helped 125 people find a place to live. Some people went to homeless shelters. A shelter is a place where homeless people can stay for a short time. Others moved into houses.</p><p>They are not sleeping in the streets anymore. The homeless people are in a better place, the governor said.</p><p>## Lots Of Families Need Homes</p><p>Many families still need a house, though. A new homeless shelter must be built soon.</p><p>Scott Morishige works with Governor Ige. He helps homeless people. A million dollars will be spent to help them, Scott said. The money will buy a new shelter. It will pay for a program, too. It is called Housing First. It will help homeless people find houses and jobs. If they have jobs, they can make money to pay for a home.</p><p>## Room For Two</p><p>Workers are building a new homeless shelter. The rooms are made from big wooden shipping boxes. The rooms fit two people.</p><p>Russ Wozniak helped to plan the shelter.</p><p>Each room has a window and a screen door. It will be nice to sit outside, Russ said.</p><p>The shelter will be done in December. People will be happy to live there.</p><p>
        """,
    ]

    texts = clean_texts(texts)
    print(texts)
    docs = get_docs(texts, nlp)
    df = get_descriptive_stats(docs)
    print(df)