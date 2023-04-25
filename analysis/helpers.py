#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Generator


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

if __name__ == '__main__':
    pass