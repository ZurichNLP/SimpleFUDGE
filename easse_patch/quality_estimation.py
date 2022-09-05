from typing import List

from tseval.feature_extraction import (
    get_compression_ratio,
    count_sentence_splits,
    get_levenshtein_similarity,
    is_exact_match,
    get_additions_proportion,
    get_deletions_proportion,
    get_wordrank_score,
    wrap_single_sentence_vectorizer,
)

from easse.utils.preprocessing import normalize


def get_average(vectorizer, orig_sentences, sys_sentences):
    cumsum = 0
    count = 0
    for orig_sentence, sys_sentence in zip(orig_sentences, sys_sentences):
        cumsum += vectorizer(orig_sentence, sys_sentence)
        count += 1
    return cumsum / count

def corpus_quality_estimation(
    orig_sentences: List[str], sys_sentences: List[str], lowercase: bool = False, tokenizer: str = '13a'
):
    orig_sentences = [normalize(sent, lowercase, tokenizer) for sent in orig_sentences]
    sys_sentences = [normalize(sent, lowercase, tokenizer) for sent in sys_sentences]
    return {
        'Compression ratio': get_average(get_compression_ratio, orig_sentences, sys_sentences),
        'Sentence splits': get_average(count_sentence_splits, orig_sentences, sys_sentences),
        'Levenshtein similarity': get_average(get_levenshtein_similarity, orig_sentences, sys_sentences),
        'Exact copies': get_average(is_exact_match, orig_sentences, sys_sentences),
        'Additions proportion': get_average(get_additions_proportion, orig_sentences, sys_sentences),
        'Deletions proportion': get_average(get_deletions_proportion, orig_sentences, sys_sentences),
        'Lexical complexity score': get_average(
            wrap_single_sentence_vectorizer(get_wordrank_score), orig_sentences, sys_sentences
        ),
    }

def get_all(vectorizer, orig_sentences, sys_sentences):
    scores = []
    for orig_sentence, sys_sentence in zip(orig_sentences, sys_sentences):
        scores.append(vectorizer(orig_sentence, sys_sentence))
    return scores

def sentence_quality_estimation(
    orig_sentences: List[str], sys_sentences: List[str], lowercase: bool = False, tokenizer: str = '13a'
):
    orig_sentences = [normalize(sent, lowercase, tokenizer) for sent in orig_sentences]
    sys_sentences = [normalize(sent, lowercase, tokenizer) for sent in sys_sentences]
    return {
        'Compression ratio': get_all(get_compression_ratio, orig_sentences, sys_sentences),
        'Sentence splits': get_all(count_sentence_splits, orig_sentences, sys_sentences),
        'Levenshtein similarity': get_all(get_levenshtein_similarity, orig_sentences, sys_sentences),
        'Exact copies': get_all(is_exact_match, orig_sentences, sys_sentences),
        'Additions proportion': get_all(get_additions_proportion, orig_sentences, sys_sentences),
        'Deletions proportion': get_all(get_deletions_proportion, orig_sentences, sys_sentences),
        'Lexical complexity score': get_all(
            wrap_single_sentence_vectorizer(get_wordrank_score), orig_sentences, sys_sentences
        ),
    }