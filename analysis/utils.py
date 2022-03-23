#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def harmonic_mean(values, coefs=None):
    """From https://github.com/facebookresearch/muss"""
    if 0 in values:
        return 0
    values = np.array(values)
    if coefs is None:
        coefs = np.ones(values.shape)
    values = np.array(values)
    coefs = np.array(coefs)
    return np.sum(coefs) / np.dot(coefs, 1 / values)

def bleu_transform(bleu):
    """From https://github.com/facebookresearch/muss"""
    min_bleu = 0
    max_bleu = 100
    bleu = max(bleu, min_bleu)
    bleu = min(bleu, max_bleu)
    return (bleu - min_bleu) / (max_bleu - min_bleu)

def sari_transform(sari):
    """From https://github.com/facebookresearch/muss"""
    min_sari = 0
    max_sari = 60
    sari = max(sari, min_sari)
    sari = min(sari, max_sari)
    return (sari - min_sari) / (max_sari - min_sari)

def fkgl_transform(fkgl):
    """From https://github.com/facebookresearch/muss"""
    min_fkgl = 0
    max_fkgl = 20
    fkgl = max(fkgl, min_fkgl)
    fkgl = min(fkgl, max_fkgl)
    return 1 - (fkgl - min_fkgl) / (max_fkgl - min_fkgl)

def combine_metrics(bleu, sari, fkgl, coefs):
    """From https://github.com/facebookresearch/muss"""
    # Combine into a score between 0 and 1, LOWER the better
    assert len(coefs) == 3
    return 1 - harmonic_mean([bleu_transform(bleu), sari_transform(sari), fkgl_transform(fkgl)], coefs)