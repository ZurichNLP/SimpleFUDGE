#!/usr/bin/env python
# coding: utf-8


"""

Author: Tannon Kew

Example Call:

    # newsela-manual parallel version sentences
    python ats_data/extract_alignments_wiki_newsela_manual.py \
        --infile resources/data/en/newsela-auto/newsela-manual/all/test.tsv \
        --corpus_dir resources/data/en/newsela_article_corpus_2016-01-29/ \
        --output_dir resources/data/en/aligned/newsela-manual_paras \
        --complex_level 0 \
        --simple_level 4 \
        --unit sent

    # newsela-manual parallel reading grade level sentences
    python ats_data/extract_aligned_paras_wiki_newsela_manual.py \
        --infile resources/data/en/newsela-auto/newsela-manual/all/test.tsv \
        --corpus_dir resources/data/en/newsela_article_corpus_2016-01-29/ \
        --output_dir resources/data/en/aligned/newsela-manual_paras \
        --complex_level 12 \
        --simple_level 3 \
        --unit sent \
        --grade_level


Notes: https://github.com/chaojiang06/wiki-auto/tree/master/wiki-manual
    
    Newsela Manual is annotated at the sentence level, as follows:

    aligned\tdinosaur-colors.en-4-20-0\tdinosaur-colors.en-3-20-0\tMoyer says that the bacteria are closer to twice the length of the impressions on the Archaeopteryx feathers.\tMoyer counters that the bacteria are closer to twice the length of the impressions on the Archaeopteryx feathers.

    "simple-sent-id and complex-sent-id are both in the format of articleIndex-level-paragraphIndex-sentIndex"

    "level" corresponds to the version of the article, which is 0 for the original version and 1,2,3 or 4 for the simplified versions.

"""

import argparse
from pathlib import Path
import re
from typing import List, Optional, Tuple
import csv
from tqdm import tqdm
import logging
import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# create a store for the file names we've seen
seen_files = set()

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--infile', type=str, required=True, help='')
    parser.add_argument('-o', '--output_dir', type=str, required=True, help='')
    parser.add_argument('--complex_level', type=int, required=False, default=0, help='')
    parser.add_argument('--simple_level', type=int, required=False, default=4, help='')
    parser.add_argument('--verbose', action='store_true', required=False, help='')
    parser.add_argument('--debug', action='store_true', required=False, help='')
    parser.add_argument('--wiki', action='store_true', required=False, help='')
    parser.add_argument('--grade_level', action='store_true', required=False, help='')
    parser.add_argument('--unit', type=str, default='sentence', required=False, help='')
    parser.add_argument('--corpus_dir', type=str, default="resources/data/en/newsela_article_corpus_2016-01-29", required=False, help='')
    return parser.parse_args()

def parse_id(id: List) -> Tuple[str, int, int, int, int]:
    """
    extracts parts from a sentence identifier used in
    newsela-manual/wiki-manual, e.g. `['chinook-recognition.en-1-0-0']`

    Note: we handle only lists of ids to simplify cases of
    m:n and 1:1 alignments
    """

    if not isinstance(id, list):
        raise RuntimeError(f'Expected as list of id(s), but got {type(id)}')

    slug, parts = id[0].split('.')
    id_parts = parts.split('-')
    if len(id_parts) == 4:
        lang, version, paragraph, sentence = id_parts
        grade = 0
    elif len(id_parts) == 5:
        lang, grade, version, paragraph, sentence = id_parts

    return slug, int(grade), int(version), int(paragraph), int(sentence)

def update_id(id: str, metadata_df: pd.DataFrame) -> str:
    """
    Matches an ID with newsela metadata to update with grade level information

    Given the ID `chinook-recognition.en-1-0-0`, 
    
    we look up the corresponding information in newsela_article_metadata.csv:
    `chinook-recognition,en,"Officially, the Chinook tribes in Washington don't exist",9.0,1,chinook-recognition.en.1.txt,test
    
    This gives us the grade level 9.0. Then the modified ID is `chinook-recognition.en-9-1-0-0`
    
    Note: newsela_article_metadata.csv should contain the following columns:
    slug, language, title, grade_level, version, filename, (split)
    """

    slug, grade, version, paragraph_idx, sentence_idx = parse_id([id])
    assert grade == 0, f'Expected grade level to be 0, but got {grade}'
    
    grade = metadata_df[(metadata_df['slug'] == slug) & (metadata_df['version'] == version)]['grade_level'].item()
    
    new_id = id.replace(f'.en-{version}', f'.en-{int(grade)}-{version}')
    
    return new_id

def get_aligned_ids(df: pd.DataFrame, cids: List, tgt_level: int = 4, grade_level: bool = False, verbose: bool = False) -> Optional[List]:
    """
    Recursive function to find alignments across all levels.
    
    Since alignments are only annotated for adjacent article versions (e.g. 0-1, 1-2, etc.), 
    we need to traverse the tree to find the aligned simple units at the target level.

    If we find aligned simple units, we return a list of their ids, otherwise None.
    """

    if not cids:
        return None
    
    # grade and version are the same for all ids in a list
    _, grade, version, _, _ = parse_id(cids)
    
    if grade_level:
        c_level = grade
    else:
        c_level = version

    if c_level == tgt_level: # found aligned simple units
        sub_df = df[df['sid'].isin(cids)]
        return sub_df.sid.to_list()
        
    else: # recursion
        sub_df = df[df['cid'].isin(cids)]
        next_cids = sub_df.sid.tolist()
        return get_aligned_ids(df, next_cids, tgt_level, grade_level, verbose)

def dedup_sents(lst: List) -> List:
    """
    Removes duplicate sentences from a set of aligned sentences keeping their original order
    """
    no_dupes = []
    [no_dupes.append(elem) for elem in lst if not no_dupes.count(elem)]    
    return no_dupes

def get_doc_text(ids: List, corpus_dir: str, verbose: bool = False) -> str:
    """
    Fetches document directly from newsela corpus.

    Note: we could just use the the file names to get aligned docs, 
    but this allows us to get aligned files according to reading levels too.
    """
    # if ids is empty, return None
    if not ids:
        return None
    
    slug, level, version, _, _ = parse_id(ids)

    filename = Path(corpus_dir) / 'articles' / f'{slug}.en.{version}.txt'

    if filename in seen_files:
        doc = None
    else:    
        with open(filename, 'r', encoding='utf8') as f:
            doc = f.read()
            # replace newlines with paragraph tags
            doc = '<p>' + doc
            doc = re.sub(r'\n+', '</p><p>', doc)
        # update seen files store so we don't read the same file twice
        seen_files.add(filename)

    return doc


def get_para_text(ids: List, corpus_dir: str, verbose: bool = False) -> str:
    """
    Fetches paragraphs directly from newsela corpus

    We match an ID with a file in the corpus dir

    Original ID: 'chinook-recognition.en-1-0-0' matches the file 'chinook-recognition.en.1.txt'
    Updated ID (to include grade level): 'chinook-recognition.en-9-1-0-0' matches the file 'chinook-recognition.en.1.txt'
    """
    
    # if ids is empty, return None
    if not ids:
        return None

    # get unique paragraph indices, since we may be dealing with sentences from different paragraphs
    paragraph_idxs = []
    for id in ids:
        slug, _, version, paragraph_idx, _ = parse_id([id])
        if paragraph_idx not in paragraph_idxs:
            paragraph_idxs.append(paragraph_idx)

    filename = Path(corpus_dir) / 'articles' / f'{slug}.en.{version}.txt'
    
    if verbose:
        logger.info(f'Fetching {id} paragraph from {filename}')
    
    paragraphs = []
    with open(filename, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip()
            if line:
                paragraphs.append(line)

    text = ' '.join([paragraphs[idx] for idx in paragraph_idxs])

    return text

def get_sent_texts(cids: List, sids: List, df: pd.DataFrame, verbose: bool = False) -> Tuple[str, str]:
    """
    Given a list of complex and simple IDs, extract the corresponding sentences
    directly from the Newsela-Manual csv file loaded as a pandas dataframe.
    """    
    c_sents = df[df['cid'].isin(cids)].csent.tolist() # extract sentences corresponding to the complex IDs
    s_sents = df[df['sid'].isin(sids)].ssent.tolist() # extract sentences corresponding to the simple IDs
    c_sents = ' '.join(dedup_sents(c_sents))
    s_sents = ' '.join(dedup_sents(s_sents))
    return c_sents, s_sents

def get_texts(
    cids: List,
    sids: List,
    df: pd.DataFrame,
    corpus_dir: str, 
    unit: str = 'sent',
    verbose: bool = False
    ) -> str:
    """
    Fetches sentences, paragraphs, or documents from the Newsela corpus according to the IDs provided.
    """
    if args.unit in ['s', 'sent', 'sentence', 'sentences']:
        c_text, s_text = get_sent_texts(cids, sids, df, verbose=args.verbose)
        
    elif args.unit in ['p', 'para', 'paragraph', 'paragraphs']:
        c_text = get_para_text(cids, args.corpus_dir, verbose=args.verbose)
        s_text = get_para_text(sids, args.corpus_dir, verbose=args.verbose)
    
    elif args.unit in ['d', 'doc', 'document', 'documents']:
        c_text = get_doc_text(cids, args.corpus_dir, verbose=args.verbose)
        s_text = get_doc_text(sids, args.corpus_dir, verbose=args.verbose)
    
    else:
        raise RuntimeError(f'Unknown unit {args.unit}')

    return c_text, s_text

def infer_output_filepath(args):
    """
    Infer output file name from args provided

    e.g. 
    
    `newsela-manual_l4-l1_test.tsv` if `--infile newsela-manual/all/test.tsv --grade_level --complex_level 4 --simple_level 1`

    `newsela-manual_v0-v4_test.tsv` if `--infile newsela-manual/all/test.tsv --complex_level 0 --simple_level 4`
    """
    
    output_file = ''
    
    if 'newsela-manual' in args.infile:
        output_file += 'newsela-manual'
    elif 'wiki-manual' in args.infile:
        output_file += 'wiki-manual'
    else:
        raise RuntimeError(f'Could not infer output file name from {infile}')
    
    if args.unit in ['s', 'sent', 'sentence', 'sentences']:
        output_file += '_sent'
    elif args.unit in ['p', 'para', 'paragraph', 'paragraphs']:
        output_file += '_para'
    elif args.unit in ['d', 'doc', 'document', 'documents']:
        output_file += '_doc'
    else:
        raise RuntimeError(f'Unknown unit {args.unit}')

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
            for src, tgt, _ in alignments:
                outf.write(f'{src}\t{tgt}\n')
        print(f'Wrote {len(alignments)} alignments to {output_filepath}')
    return

def extract_alignments_from_newsela(args):
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

    df = pd.read_csv(args.infile, sep='\t', header=None, names=['label', 'sid', 'cid', 'ssent', 'csent'], quoting=csv.QUOTE_NONE)
    if args.verbose:
        logging.info(f'DF contains {len(df)} items')
        
    df = df[df['label'] != 'notAligned'] # filter all not aligned sentences
    if args.verbose:
        logging.info(f'Removed `notAligned`. DF contains {len(df)} items')

    # if args.grade_level and args.metadata_file is not None:
    metadata_file = Path(args.corpus_dir) / 'articles_metadata.csv'
    metadata_df = pd.read_csv(metadata_file, sep=',', header=0, quoting=csv.QUOTE_MINIMAL)
    
    # update ids with to include the grade level
    # note, we need to keep the version to match the with files in the newsela corpus
    df['cid'] = df['cid'].apply(lambda x: update_id(x, metadata_df))
    df['sid'] = df['sid'].apply(lambda x: update_id(x, metadata_df))

    # collect all nodes at the target complex level as roots
    root_nodes = []
    for id in df['cid'].unique():
        slug, grade, version, paragraph_idx, sentence_idx = parse_id([id])

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
        
        # print(root_node)
        
        slug, _, _, _, _ = parse_id(root_node)
        
        tgt_ids = get_aligned_ids(df, root_node, args.simple_level, args.grade_level, args.verbose)

        # c_text = get_text(root_node, tgt_ids, df, args.corpus_dir, args.unit, args.verbose)
        if tgt_ids is None:
            continue

        c_text, s_text = get_texts(root_node, tgt_ids, df, args.corpus_dir, args.unit, args.verbose)
    
        if c_text and s_text:
            alignments.append((c_text, s_text, slug))
                 
    alignments = dedup_sents(alignments)

    return alignments


# def extract_alignments_from_wiki(args, max_sim=0.6):
#     """
#     Processes annotated alignment file from Wiki-Manual (e.g. `wiki-auto/wiki-manual/test.tsv`)
#     """
    
#     df = pd.read_csv(args.infile, sep='\t', header=None, names=['label', 'sid', 'cid', 'ssent', 'csent', 'gleu_score'], quoting=csv.QUOTE_NONE)
#     if args.verbose:
#         print(f'DF contains {len(df)} items')

#     df = df[df['label'] != 'notAligned'] # filter all not aligned sentences
#     if args.verbose:
#         print(f'Removed `notAligned`. DF contains {len(df)} items')

#     df = df[df['gleu_score'] < max_sim] # filter all aligned sentence that are too similar
#     if args.verbose:
#         print(f'Removed items with sim above `{max_sim}`. DF contains {len(df)} items')


#     root_nodes = [[id] for id in df['cid'].unique()]

#     if args.verbose:
#         print(len(root_nodes))
#         print(root_nodes[:5], '...')

#     # collect alignments
#     alignments = []
#     for root_node in root_nodes:
#         sub_df = df[(df['cid'].isin(root_node))]
#         csents = sub_df.csent.tolist()
#         ssents = sub_df.ssent.tolist()
#         try:
#             src = ' '.join(csents).strip()
#         except TypeError:
#             print(f'Could not parse sentence {root_node} as string. Check for unclosed quotations!', csents)
#             continue
            
#         try:
#             tgt = ' '.join(ssents).strip()
#         except TypeError:
#             print(f'Could not parse sentence {root_node} as string. Check for unclosed quotations!', csents)
#             continue

#         if src and tgt:
#             alignments.append((src, tgt, root_node[0]))
                
#     # write collected alignments to outfile
#     if len(alignments) > 0:
#         if args.outfile:
#             with open(args.outfile, 'w', encoding='utf8') as outf:
#                 for src, tgt, _ in alignments:
#                     outf.write(f'{src}\t{tgt}\n')

#         print(f'Finished writing {len(alignments)} alignments to {args.outfile}')
#     else:
#         print(f'No alignments written to {args.outfile}')

if __name__ == '__main__':

    args = set_args()
    
    if args.debug:
        test()
    else:
        if args.wiki:
            raise NotImplementedError(
                'Wiki alignments not yet implemented.' \
                'Use older version of this script ' \
                '(simple_fudge/ats_data/extract_aligned_sents_wiki_newsela_manual.py)'
            )
            aligned_paras = extract_alignments_from_wiki(args)
        else:
            aligned_paras = extract_alignments_from_newsela(args)

    # write collected alignments to outfile
    write_to_file(aligned_paras, args)