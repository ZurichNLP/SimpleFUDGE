#!/usr/bin/env python
# coding: utf-8

"""

Author: Tannon Kew

Example Call:

    # newsela-auto
    python data_prep/extract_alignments_newsela_auto.py \
        --infile resources/data/en/newsela-auto/newsela-auto/all_data/aligned-sentence-pairs-all.tsv \
        --outfile resources/data/en/aligned/newsela_auto_v0_v4_train_v2.tsv \
        --complex_level 0 --simple_level 4

Note: Newsela-Auto (newsela-auto/all_data/aligned-sentence-pairs-all.tsv) does NOT handle M:N alignments.
Partial alignments are treated as regular alignments!

"""

import argparse
import csv
import pandas as pd
import logging
from tqdm import tqdm
tqdm.pandas()
import warnings
from typing import List
from pathlib import Path
from .annotate_newsela_splits import (
    newsela_manual_train_article_slugs, 
    newsela_manual_dev_article_slugs, 
    newsela_manual_test_article_slugs,
) 

from .extract_alignments_newsela_manual import (
    parse_id,
    update_id,
    get_aligned_ids,
    get_texts,
    dedup_sents
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--infile', type=str, required=True, help='')
    parser.add_argument('-o', '--output_dir', type=str, required=True, help='')
    parser.add_argument('--complex_level', type=int, required=False, default=0, help='')
    parser.add_argument('--simple_level', type=int, required=False, default=4, help='')
    parser.add_argument('--verbose', action='store_true', required=False, help='')
    parser.add_argument('--wiki', action='store_true', required=False, help='')
    parser.add_argument('--grade_level', action='store_true', required=False, help='')
    parser.add_argument('--unit', type=str, default='sentence', required=False, help='')
    parser.add_argument('--corpus_dir', type=str, default="resources/data/en/newsela_article_corpus_2016-01-29", required=False, help='')


    return parser.parse_args()

def dedup_paragraphs(alignments):
    """
    Deduplicate alignments at the paragraph level.
    Compares the first element of each tuple in the list of alignments and keeps only the one with the longest second element.    
    """
    
    deduped_alignments = []
    # Sort the list of tuples by the first element (ascending) and by the length of the second element (descending)
    alignments.sort(key=lambda x: (x[0], -len(x[1])))
    # Alternatively, sort by the first element (ascending) and by the absolute difference in length of the second element (ascending)
    # alignments.sort(key=lambda x: (x[0], abs(len(x[1]) - len(x[0]))))

    # Iterate through the sorted list of tuples
    for i, t in enumerate(alignments):
        # If it's the first tuple in the list OR if the first element of the current tuple is different
        # from the first element of the previous tuple, append the tuple to the result list.
        if i == 0 or t[0] != alignments[i - 1][0]:
            deduped_alignments.append(t)
    return deduped_alignments


def extract_alignments_from_newsela_auto(args):
    """
    Processes annotated alignment file from Newsela-Manual (e.g. `newsela-auto/newsela-manual/all/test.tsv`)
    """

    if args.grade_level:
        if args.corpus_dir is None:
            raise RuntimeError(f'If `--grade_level` is set, `--corpus_dir` containing Newsela must be provided')
        if args.complex_level == 0:
            raise RuntimeError(f'If `--grade_level` is set, `--complex_level` must correspond to a valid reading grade level in Newsela (e.g [1-12])')
        if args.simple_level == 0:
            raise RuntimeError(f'If `--grade_level` is set, `--simple_level` must correspond to a valid reading grade level in Newsela (e.g [1-12])')
    else:
        if args.complex_level not in [0, 1, 2, 3]:
            raise RuntimeError(f'If `--grade_level` is not set, `--complex_level` must correspond to a valid article version in Newsela (e.g [0-4])')
        if args.simple_level not in [1, 2, 3, 4]:
            raise RuntimeError(f'If `--grade_level` is not set, `--simple_level` must correspond to a valid article version in Newsela (e.g [0-4])')

    # we persist the data frame to disk so that we don't have to recompute it every time
    tmp_data_frame = Path(args.infile).parent / f'{Path(args.infile).stem}_updated_ids.tsv'
    if tmp_data_frame.exists():
        print(f'Loading data frame from {tmp_data_frame}')
        df = pd.read_csv(tmp_data_frame, sep='\t', header=None, names=['sid', 'ssent', 'cid', 'csent'], quoting=csv.QUOTE_NONE)
    else:
        df = pd.read_csv(args.infile, sep='\t', header=None, names=['sid', 'ssent', 'cid', 'csent'], quoting=csv.QUOTE_NONE)
        if args.verbose:
            logging.info(f'DF contains {len(df)} items')
        
        # if args.grade_level and args.metadata_file is not None:
        metadata_file = Path(args.corpus_dir) / 'articles_metadata.csv'
        metadata_df = pd.read_csv(metadata_file, sep=',', header=0, quoting=csv.QUOTE_MINIMAL)
        
        # update ids with to include the grade level
        # note, we need to keep the version info to match the with files in the newsela corpus
        df['cid'], df['sid'] = zip(*df.progress_apply(lambda row: (update_id(row['cid'], metadata_df), update_id(row['sid'], metadata_df)), axis=1))
        # df['cid'] = df['cid'].progress_apply(lambda x: update_id(x, metadata_df))
        # df['sid'] = df['sid'].progress_apply(lambda x: update_id(x, metadata_df))

        # save updated alignments
        df.to_csv(str(tmp_data_frame), sep='\t', header=False, index=False, quoting=csv.QUOTE_NONE)
        print(f'Saved updated alignments to {tmp_data_frame}')

    # collect all nodes at the target complex level as roots
    root_nodes = []
    for id in df['cid'].unique():
        _, grade, version, _, _ = parse_id([id])

        if args.grade_level:
            c_level = grade
        else:
            c_level = version

        if c_level == args.complex_level:
            root_nodes.append([id])

    if args.verbose:
        logging.info(len(root_nodes))
        logging.info(root_nodes[:5], '...')

    # collect alignments
    alignments = []
    for root_node in tqdm(root_nodes, total=len(root_nodes)):
        
        slug, _, _, _, _ = parse_id(root_node)
        tgt_ids_ = get_aligned_ids(df, root_node, args.simple_level, args.grade_level, args.verbose)
        if tgt_ids_ is None:
            continue
        # print(root_node, tgt_ids_)
        
        tgt_ids = []
        for tgt_id in sorted(tgt_ids_):
            _, grade, version, _, _ = parse_id([tgt_id])
            if args.grade_level:
                s_level = grade
            else:
                s_level = version
            if s_level == args.simple_level and tgt_id not in tgt_ids:
                tgt_ids.append(tgt_id)
        
        if tgt_ids is None:
            continue
        # print(root_node, tgt_ids)
        
        c_text, s_text = get_texts(root_node, tgt_ids, df, args.corpus_dir, args.unit, args.verbose)
        
        if c_text and s_text:
            alignments.append((c_text, s_text, slug))
                 
    alignments = dedup_sents(alignments)

    # NOTE: alignments are at the sentence level, so we need to do some extra work to extract the paragraph alignments
    # however, multiple sentences from one paragraph may be aligned to a single sentence in the other paragraph,
    # while other sentences in the paragraph are aligned to other sentences in other paragraphs.
    if args.unit in ['p', 'para', 'paras', 'paragraph', 'paragraphs']:
        alignments = dedup_paragraphs(alignments)

    return alignments

def infer_output_filepath(args):
    """
    Infer output file name from args provided

    e.g. 
    
    `newsela-manual_l4-l1_test.tsv` if `--infile newsela-manual/all/test.tsv --grade_level --complex_level 4 --simple_level 1`

    `newsela-manual_v0-v4_test.tsv` if `--infile newsela-manual/all/test.tsv --complex_level 0 --simple_level 4`
    """

    # infer sub dir for output files based on dataset, unit and grade vs level
    dir_name = ''
    
    if 'newsela-auto' in args.infile:
        dir_name += 'newsela_auto'
    elif 'wiki-auto' in args.infile:
        dir_name += 'wiki_auto'
    else:
        raise RuntimeError(f'Could not infer output file name from {args.infile}')
    
    if args.unit in ['s', 'sent', 'sents', 'sentence', 'sentences']:
        dir_name += '_sents'
    elif args.unit in ['p', 'para', 'paras', 'paragraph', 'paragraphs']:
        dir_name += '_paras'
    elif args.unit in ['d', 'doc', 'docs', 'document', 'documents']:
        dir_name += '_docs'
    else:
        raise RuntimeError(f'Unknown unit {args.unit}')

    # infer output file name based on complex-simple version/level and split
    if args.grade_level:
        dir_name += '_level'
    else:
        dir_name += '_version'
    
    split = Path(args.infile).stem
    output_file = f'{args.complex_level}-{args.simple_level}_{split}.tsv'
            
    # build output file path
    output_filepath = Path(args.output_dir) / dir_name / output_file
    
    return output_filepath

def write_to_file(alignments, args):

    if len(alignments) == 0:
        logger.info(f'No alignments found!')
    else:
        output_filepath = infer_output_filepath(args)
        output_filepath.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f'Writing to {output_filepath}')
        with open(output_filepath, 'w', encoding='utf8') as outf:
            for src, tgt, _ in alignments:
                outf.write(f'{src}\t{tgt}\n')
        print(f'Wrote {len(alignments)} alignments to {output_filepath}')
    return

if __name__ == '__main__':

    args = set_args()
    
    aligned_paras = extract_alignments_from_newsela_auto(args)

    # write collected alignments to outfile
    write_to_file(aligned_paras, args)