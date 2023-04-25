#!/usr/bin/env python
# coding: utf-8


"""

Author: Tannon Kew

Example Call:

    # newsela-manual parallel version sentences
    python data_prep/extract_alignments_newsela.py \
        --infile resources/data/en/newsela-auto/newsela-manual/all/test.tsv \
        --corpus_dir resources/data/en/newsela_article_corpus_2016-01-29/ \
        --output_dir resources/data/en/aligned \
        --complex_level 0 \
        --simple_level 4 \
        --unit sent

    # newsela-manual parallel reading grade level sentences
    python data_prep/extract_alignments_newsela.py \
        --infile resources/data/en/newsela-auto/newsela-manual/all/test.tsv \
        --corpus_dir resources/data/en/newsela_article_corpus_2016-01-29/ \
        --output_dir resources/data/en/aligned \
        --complex_level 12 \
        --simple_level 3 \
        --unit sent \
        --grade_level

    # newsela-auto parallel version sentences
    python data_prep/extract_alignments_newsela.py \
        --infile resources/data/en/newsela-auto/newsela-auto/all_data/aligned-sentence-pairs-all.tsv \
        --corpus_dir resources/data/en/newsela_article_corpus_2016-01-29/ \
        --output_dir resources/data/en/aligned \
        --complex_level 0 \
        --simple_level 4 \
        --unit sent
        


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
tqdm.pandas()
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
    parser.add_argument('--grade_level', action='store_true', required=False, help='If set, extract alignments for grade level sentences, otherwise, extract alignments for parallel versions.')
    parser.add_argument('--unit', type=str, default='sentence', required=False, help='Unit to extract alignments for. Options: sentence, paragraph, document')
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

def dedup_paragraphs(alignments):
    """
    Deduplicate alignments at the paragraph level.
    Compares the first element of each tuple in the list of alignments and keeps only the one with the longest second element.    
    """
    
    deduped_alignments = []
    # Sort the list of tuples by the first element (ascending) and by the length of the second element (descending)
    alignments.sort(key=lambda x: (x[0], -len(x[1])))
    # Alternatively, could also sort by the first element (ascending) and by the absolute difference in length of the second element (ascending)
    # alignments.sort(key=lambda x: (x[0], abs(len(x[1]) - len(x[0]))))

    # Iterate through the sorted list of tuples
    for i, t in enumerate(alignments):
        # If it's the first tuple in the list OR if the first element of the current tuple is different
        # from the first element of the previous tuple, append the tuple to the result list.
        if i == 0 or t[0] != alignments[i - 1][0]:
            deduped_alignments.append(t)
    return deduped_alignments

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
            doc = doc + '</p>'
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
    if unit in ['s', 'sent', 'sents', 'sentence', 'sentences']:
        c_text, s_text = get_sent_texts(cids, sids, df, verbose=verbose)
        
    elif unit in ['p', 'para', 'paras', 'paragraph', 'paragraphs']:
        c_text = get_para_text(cids, corpus_dir, verbose=verbose)
        s_text = get_para_text(sids, corpus_dir, verbose=verbose)
    
    elif unit in ['d', 'doc', 'docs', 'document', 'documents']:
        c_text = get_doc_text(cids, corpus_dir, verbose=verbose)
        s_text = get_doc_text(sids, corpus_dir, verbose=verbose)
    
    else:
        raise RuntimeError(f'Unknown unit {unit}')

    return c_text, s_text

def infer_output_filepath(args):
    """
    Infer output file name from args provided

    e.g. 
    
    `newsela-manual_l4-l1_test.tsv` if `--infile newsela-manual/all/test.tsv --grade_level --complex_level 4 --simple_level 1`

    `newsela-manual_v0-v4_test.tsv` if `--infile newsela-manual/all/test.tsv --complex_level 0 --simple_level 4`
    """

    # infer sub dir for output files based on dataset, unit and grade vs level
    dir_name = ''
    
    if 'newsela-auto/newsela-manual' in args.infile: # e.g. resources/data/en/newsela-auto/newsela-manual/all/dev.tsv
        dir_name += 'newsela_manual'
    elif 'wiki-auto/wiki-manual' in args.infile:
        dir_name += 'wiki_manual'
    elif 'newsela-auto/newsela-auto' in args.infile: # e.g. resources/data/en/newsela-auto/newsela-auto/all_data/aligned-sentence-pairs-all.tsv
        dir_name += 'newsela_auto'
    elif 'wiki-auto/wiki-auto' in args.infile:
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
    if split in ['train', 'dev', 'test']:
        output_file = f'{args.complex_level}-{args.simple_level}_{split}.tsv'
    else:
        output_file = f'{args.complex_level}-{args.simple_level}.tsv'
            
    # build output file path
    output_filepath = Path(args.output_dir) / dir_name / output_file
    
    return output_filepath

def write_to_file(alignments, output_filepath):

    if len(alignments) == 0:
        logger.info(f'No alignments found!')
    else:
        logger.info(f'Writing to {output_filepath}')
        with open(output_filepath, 'w', encoding='utf8') as outf:
            for src, tgt, _ in alignments:
                outf.write(f'{src}\t{tgt}\n')
        print(f'Wrote {len(alignments)} alignments to {output_filepath}')
    return

def extract_alignments_from_newsela_manual(args):
    """
    Processes manually-annotated alignment file from Newsela-Manual (e.g. `newsela-auto/newsela-manual/all/test.tsv`)
    """

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
    # note, we need to keep the version info to match the with files in the newsela corpus
    df['cid'], df['sid'] = zip(*df.progress_apply(lambda row: (update_id(row['cid'], metadata_df), update_id(row['sid'], metadata_df)), axis=1))

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


def extract_alignments_from_newsela_auto(args):
    """
    Similar to the above function but for the auto-aligned data, which is annotated differently.

    Processes auto-annotated alignment file from Newsela-Auto (e.g. `newsela-auto/newsela-auto/all_data/aligned-sentence-pairs-all.tsv`)
    """

    # if an updated dataframe already exists, load it
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

        # persist the updated data frame to disk so that we don't have to recompute it every time
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

def check_args(args):
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

    return

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
    
    check_args(args)

    output_filepath = infer_output_filepath(args)
    output_filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if 'newsela_auto' in str(output_filepath):
        alignments = extract_alignments_from_newsela_auto(args)
    elif 'newsela_manual' in str(output_filepath):
        alignments = extract_alignments_from_newsela_manual(args)
    
    # write collected alignments to outfile
    write_to_file(alignments, output_filepath=output_filepath)