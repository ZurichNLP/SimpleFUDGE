#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __Author__ = 'Tannon Kew'
# __Email__ = 'kew@cl.uzh.ch
# __Date__ = '2023-03-03'


"""

Generic helper functions for file handling

"""

import re
import json
import logging
import hashlib
import pprint
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Generator

logger = logging.getLogger(__name__)

def iter_text_lines(file: Union[str, Path]) -> Generator[str, None, None]:
    """Generator that yields lines from a regular text file."""
    with open(file, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            else:
                yield line

def iter_json_lines(file: Union[str, Path]) -> Generator[Dict, None, None]:
    """Fetch dictionary-object lines from a JSONL file"""
    with open(file, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            else:
                yield json.loads(line)

def iter_split_lines(file: Union[str, Path], delimiter: str = '\t', src_key: str = 'complex', tgt_key: str = 'simple') -> Generator[Dict, None, None]:
    """Fetch dictionary-object lines from a TSV file"""
    with open(file, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip().split(delimiter)
            if len(line) == 0:
                continue
            line_d = {src_key: line[0], tgt_key: line[1:]}
            yield line_d

def iter_lines(file: Union[str, Path]) -> Generator[Union[str, Dict], None, None]:
    """Wraps `iter_text_lines` and `iter_json_lines` to fetch lines from file"""
    if str(file).endswith(".jsonl") or str(file).endswith(".json"):
        return iter_json_lines(file)
    elif str(file).endswith(".tsv"):
        return iter_split_lines(file, delimiter='\t')
    else:
        return iter_text_lines(file)

def load_few_shot_prompts(fsp_file: Union[str, Path]) -> List[str]:
    """Returns a list of few-shot prompts."""
    fsprompts = [l for l in iter_lines(fsp_file)]
    logger.info(f"Loaded {len(fsprompts)} few-shot prompts")
    return fsprompts

def load_prompts(p_file: Union[str, Path]) -> List[str]:
    """Returns a list of prompts."""
    prompts = [l for l in iter_lines(p_file)]
    logger.info(f"Loaded {len(prompts)} prompts")
    return prompts

def merge_prompts(prompts: List[str], fsprompts: Optional[List[str]] = None) -> List[str]:
    if not fsprompts: # zero-shot setting
        return prompts
    elif len(fsprompts) == 1: # every input is concatenated with the same fsprompt
        return [fsprompts[0] + prompt for prompt in prompts]
    else:
        if len(fsprompts) != len(prompts):
            raise RuntimeError("Number of few-shot prompts must be 1 or equal to number of prompts!")
        return [fsprompts[i] + prompts[i] for i in range(len(prompts))]

def iter_json_batches(file: str, batch_size: int = 3) -> Generator[List[Dict], None, None]:
    """Fetch batched lines from jsonl file"""
    current_batch = []
    c = 0
    for line in iter_json_lines(file):
        current_batch.append(line)
        c += 1
        if c == batch_size and len(current_batch) > 0:
            yield current_batch
            # reset vars for next batch
            c = 0
            current_batch = []    
    if len(current_batch) > 0:
        yield current_batch # don't forget the last one!

def iter_text_batches(file: Union[str, Path], batch_size: int = 3) -> Generator[List[str], None, None]:
    """Fetch batched lines from file"""
    current_batch = []
    c = 0
    for line in iter_lines(file):
        current_batch.append(line)
        c += 1
        if c == batch_size and len(current_batch) > 0:
            yield current_batch
            # reset vars for next batch
            c = 0
            current_batch = []    
    if len(current_batch) > 0:
        yield current_batch # don't forget the last one!

def iter_batches(file: Union[str, Path], batch_size: int = 3) -> Generator[Union[List[str], List[Dict]], None, None]:
    """Wraps `iter_text_batches` and `iter_json_batches` to fetch batched lines from file"""
    if str(file).endswith(".jsonl") or str(file).endswith(".json"):
        return iter_json_batches(file, batch_size)
    else:
        return iter_text_batches(file, batch_size)


if __name__ == "__main__":
    pass