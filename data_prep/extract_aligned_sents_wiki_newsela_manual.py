#!/usr/bin/env python
# coding: utf-8


"""

Author: Tannon Kew

# NOTE: This script is deprecated. Use `extract_alignments_wiki_newsela_manual.py` instead.

Example Call:

    # newsela
    python extract_aligned_sents_wiki_newsela_manual.py \
        --infile /srv/scratch1/kew/ats/data/en/newsela-auto/newsela-manual/all/test.tsv \
        --outfile /srv/scratch1/kew/ats/data/en/aligned/newsela_manual_v0_v4_test.tsv \
        --complex_level 0 \
        --simple_level 4
    
    # wiki-manual
    python extract_aligned_sents_wiki_newsela_manual.py \
        --infile /srv/scratch1/kew/ats/data/en/wiki-auto/wiki-manual/test.tsv \
        --outfile /srv/scratch1/kew/ats/data/en/aligned/wiki_manual_test.tsv \
        --complex_level 0 \
        --simple_level 1 --wiki

    # newsela-manual reading grade levels
    python extract_aligned_sents_wiki_newsela_manual.py \
        --infile /srv/scratch1/kew/ats/data/en/newsela-auto/newsela-manual/all/test.tsv \
        --outfile /srv/scratch1/kew/ats/data/en/aligned/newsela_manual_12_3_test.tsv \
        --complex_level 12 \
        --simple_level 3 \
        --metadata_file data_prep/newsela_articles_metadata_with_splits.csv \
        --grade_level


Notes: https://github.com/chaojiang06/wiki-auto/tree/master/wiki-manual
    
    Newsela Manual is annotated at the sentence level, as follows:

    aligned\tdinosaur-colors.en-4-20-0\tdinosaur-colors.en-3-20-0\tMoyer says that the bacteria are closer to twice the length of the impressions on the Archaeopteryx feathers.\tMoyer counters that the bacteria are closer to twice the length of the impressions on the Archaeopteryx feathers.

    "simple-sent-id and complex-sent-id are both in the format of articleIndex-level-paragraphIndex-sentIndex"

    "level" corresponds to the version of the article, which is 0 for the original version and 1,2,3 or 4 for the simplified versions.

"""

import argparse
from bdb import set_trace
from typing import List
import csv
from tqdm import tqdm

import pandas as pd

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--infile', type=str, required=True, help='')
    parser.add_argument('-o', '--outfile', type=str, required=True, help='')
    parser.add_argument('--complex_level', type=int, required=False, default=0, help='')
    parser.add_argument('--simple_level', type=int, required=False, default=4, help='')
    parser.add_argument('--verbose', action='store_true', required=False, help='')
    parser.add_argument('--debug', action='store_true', required=False, help='')
    parser.add_argument('--wiki', action='store_true', required=False, help='')
    parser.add_argument('--grade_level', action='store_true', required=False, help='')
    parser.add_argument('--metadata_file', type=str, default="newsela_articles_metadata_with_splits.csv", required=False, help='')
    return parser.parse_args()

def get_slug_from_full_id(id: List):
    """
    extracts slug (unique name id) from a sentence identifier used in
    newsela-manual/wiki-manual, e.g. `['chinook-recognition.en-1-0-0']`

    Note: we handle only lists of ids to simplify cases of
    m:n and 1:1 alignments
    """
    if not isinstance(id, list):
        raise RuntimeError(f'Expected as list of id(s), but got {type(id)}')
    slug, id = id[0].split('.')
    return slug
    
def get_level_from_full_id(id: List):
    """
    extracts simplification level (version) from a sentence identifier used in
    newsela-manual/wiki-manual, e.g. `['chinook-recognition.en-1-0-0']`

    Note: we handle only lists of ids to simplify cases of
    m:n and 1:1 alignments
    """
    if not isinstance(id, list):
        raise RuntimeError(f'Expected as list of id(s), but got {type(id)}')
    slug, id = id[0].split('.')
    return int(id.split('-')[1])

def get_grade_level_id(id, metadata_df):
    """
    matches a sid/cid with newsela metadata to get grade level

    e.g. the sid `chinook-recognition.en-1-0-0` matches the article `chinook-recognition,en,"Officially, the Chinook tribes in Washington don't exist",9.0,1,chinook-recognition.en.1.txt,test
    
    newsela_article_metadata.csv contains the following columns:
    slug, language, title, grade_level, version, filename, split
    """

    slug = get_slug_from_full_id([id])
    version = get_level_from_full_id([id])
    level = metadata_df[(metadata_df['slug'] == slug) & (metadata_df['version'] == version)]['grade_level'].item()
    new_id = id.replace(f'.en-{version}', f'.en-{int(level)}')
    return new_id

def extract_pairs(df, cid: List, tgt_level: int = 4):
    """
    Recursive function to extract aligned sentences from the
    next adjacent simplification level
    """
    if not cid: 
        return None
    
    c_level = get_level_from_full_id(cid)
    if c_level == tgt_level: # found aligned simple units
        sub_df = df[df['sid'].isin(cid)]
        sents = ' '.join(dedup_sents(sub_df.ssent.tolist()))
        return sents
        
    else: # recursion
        sub_df = df[df['cid'].isin(cid)]
        next_cid = sub_df.sid.tolist()
        return extract_pairs(df, next_cid, tgt_level)

def dedup_sents(lst):
    """
    Removes duplicate sentences from a set of aligned sentences keeping order
    """
    no_dupes = []
    [no_dupes.append(elem) for elem in lst if not no_dupes.count(elem)]    
    return no_dupes

def test():
    df = pd.read_csv(args.infile, sep='\t', header=None, names=['label', 'sid', 'cid', 'ssent', 'csent'], quoting=csv.QUOTE_NONE)
    df = df[df['label'] != 'notAligned'] # filter all not aligned sentences
    print(extract_pairs(df, ['chinook-recognition.en-0-0-0']))
    print(extract_pairs(df, ['chinook-recognition.en-0-0-1']))
    print(extract_pairs(df, ['chinook-recognition.en-0-0-2']))

def infer_output_filepath(args):
    """
    Infer output file name from args provided

    e.g. 
    
    `newsela-manual_sents_l4-l1_test.tsv` if `--infile .../newsela-manual/... --grade_level --complex_level 4 --simple_level 1`

    `newsela-manual_sents_v0-v4_test.tsv` if `--infile .../newsela-manual/... --complex_level 0 --simple_level 4`
    """
    
    output_file = ''
    
    if 'newsela-manual' in args.infile:
        output_file += 'newsela-manual_sents'
    elif 'wiki-manual' in args.infile:
        output_file += 'wiki-manual_sents'
    else:
        raise RuntimeError(f'Could not infer output file name from {infile}')
    
    if args.grade_level:
        output_file += f'_l{args.complex_level}-l{args.simple_level}'
    else:
        output_file += f'_v{args.complex_level}-v{args.simple_level}'

    split = Path(args.infile).stem

    output_file += f'_{split}.tsv'

    output_filepath = Path(args.output_dir) / output_file
    
    return output_filepath

def write_to_file(alignments, args):

    if len(alignments) == 0:
        logger.info(f'No alignments found!')
    else:
        output_filepath = infer_output_filepath(args)
        output_filepath.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f'Writing to {output_filepath}')
        with open(output_filepath, 'w', encoding='utf8') as outf:
            for src, tgt, slug in alignments:
                outf.write(f'{src}\t{tgt}\t{slug}\n')
        print(f'Wrote {len(alignments)} alignments to {output_filepath}')
    return

def parse_newsela_data(args):
    """
    Processes annotated alignment file from Newsela-Manual (e.g. `newsela-auto/newsela-manual/all/test.tsv`)
    """

    df = pd.read_csv(args.infile, sep='\t', header=None, names=['label', 'sid', 'cid', 'ssent', 'csent'], quoting=csv.QUOTE_NONE)
    if args.verbose:
        print(f'DF contains {len(df)} items')
        
    df = df[df['label'] != 'notAligned'] # filter all not aligned sentences
    if args.verbose:
        print(f'Removed `notAligned`. DF contains {len(df)} items')

    if args.grade_level and args.metadata_file is not None:
        metadata_df = pd.read_csv(args.metadata_file, sep=',', header=0, quoting=csv.QUOTE_MINIMAL)
        # update ids with grade level instead of version number
        df['cid'] = df['cid'].apply(lambda x: get_grade_level_id(x, metadata_df))
        df['sid'] = df['sid'].apply(lambda x: get_grade_level_id(x, metadata_df))

    root_nodes = [[id] for id in df['cid'].unique() if get_level_from_full_id([id]) == args.complex_level]
    
    if args.verbose:
        print(len(root_nodes))
        print(root_nodes[:5], '...')

    # collect alignments
    alignments = []
    for root_node in tqdm(root_nodes, total=len(root_nodes)):
        sub_df = df[(df['cid'].isin(root_node))] 
        
        csents = dedup_sents(sub_df.csent.tolist())
        if len(set(csents)) != len(csents):
            raise RuntimeError
        try:
            src = ' '.join(csents)
        except TypeError:
            print(f'Could not parse sentence {root_node} as string. Check for unclosed quotations!', csents)
            continue
            
        tgt = extract_pairs(df, root_node, args.simple_level)
        if tgt:
            alignments.append((src, tgt, get_slug_from_full_id(root_node)))
         
    # write collected alignments to outfile
    if len(alignments) > 0:
        if args.outfile:
            with open(args.outfile, 'w', encoding='utf8') as outf:
                for src, tgt, _ in alignments:
                    outf.write(f'{src}\t{tgt}\n')

        print(f'Finished writing {len(alignments)} alignments to {args.outfile}')
    else:
        print(f'No alignments written to {args.outfile}')

def parse_wiki_data(args, max_sim=0.6):
    """
    Processes annotated alignment file from Wiki-Manual (e.g. `wiki-auto/wiki-manual/test.tsv`)
    """
    
    df = pd.read_csv(args.infile, sep='\t', header=None, names=['label', 'sid', 'cid', 'ssent', 'csent', 'gleu_score'], quoting=csv.QUOTE_NONE)
    if args.verbose:
        print(f'DF contains {len(df)} items')

    df = df[df['label'] != 'notAligned'] # filter all not aligned sentences
    if args.verbose:
        print(f'Removed `notAligned`. DF contains {len(df)} items')

    df = df[df['gleu_score'] < max_sim] # filter all aligned sentence that are too similar
    if args.verbose:
        print(f'Removed items with sim above `{max_sim}`. DF contains {len(df)} items')


    root_nodes = [[id] for id in df['cid'].unique()]

    if args.verbose:
        print(len(root_nodes))
        print(root_nodes[:5], '...')

    # collect alignments
    alignments = []
    for root_node in root_nodes:
        sub_df = df[(df['cid'].isin(root_node))]
        csents = sub_df.csent.tolist()
        ssents = sub_df.ssent.tolist()
        try:
            src = ' '.join(csents).strip()
        except TypeError:
            print(f'Could not parse sentence {root_node} as string. Check for unclosed quotations!', csents)
            continue
            
        try:
            tgt = ' '.join(ssents).strip()
        except TypeError:
            print(f'Could not parse sentence {root_node} as string. Check for unclosed quotations!', csents)
            continue

        if src and tgt:
            alignments.append((src, tgt, root_node[0]))
                
    # write collected alignments to outfile
    if len(alignments) > 0:
        if args.outfile:
            with open(args.outfile, 'w', encoding='utf8') as outf:
                for src, tgt, _ in alignments:
                    outf.write(f'{src}\t{tgt}\n')

        print(f'Finished writing {len(alignments)} alignments to {args.outfile}')
    else:
        print(f'No alignments written to {args.outfile}')

if __name__ == '__main__':

    args = set_args()
    
    if args.debug:
        test()
    else:
        if args.wiki:
            parse_wiki_data(args)
        else:
            parse_newsela_data(args)