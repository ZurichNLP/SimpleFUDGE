#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Example call:

    python data_prep/prepare_data_for_unieval.py \
        --pos_examples resources/data/en/aligned/newsela_manual_sents_version/0-4_dev.tsv \
        --neg_examples resources/uni_eval/muss/0-4_dev_lr1.0_ls0.9_wr1.0_td1.0.txt \
        --output_dir resources/uni_eval/

    python data_prep/prepare_data_for_unieval.py \
        --pos_examples resources/data/en/aligned/newsela_auto_sents_version/0-4.tsv \
        --neg_examples resources/uni_eval/muss/0-4_lr1.0_ls0.9_wr1.0_td1.0.txt \
        --output_dir resources/uni_eval/

"""

import argparse
from pathlib import Path
import warnings
import pandas as pd
from tqdm import tqdm
import csv
import json
import random
from easse import fkgl, quality_estimation # samsa fails dep: tupa
from evaluate import load

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pos_examples', type=str, required=True, help='line-aligned parallel pairs of source and human-written reference')
    parser.add_argument('--neg_examples', type=str, required=True, help='corrupted examples')
    parser.add_argument('--output_dir', type=str, required=True, help='output directory, outfile name is inferred from input file names')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--num_samples', type=int, default=30000, help='number of samples to take from the positive/negative examples')
    return parser.parse_args()

def iter_lines(path, sep='\t', col_num=0):
    with open(path) as f:
        for line in f:
            line = line.strip().split(sep)
            if len(line) < col_num + 1:
                raise ValueError(f'Line {line} has less than {col_num} columns')
            if not line[col_num]:
                warnings.warn(f'Line {line} has empty value in column {col_num}')
            yield line[col_num].strip()

def clean_parallel_pairs(df, verbose=True):
    """
    remove duplicates, empty strings, and exact matches from parallel pairs
    """
    if verbose:
        print(f'Initial number of parallel pairs: {len(df)}')

    df = df.drop_duplicates(subset=['source', 'pos', 'neg'], keep='first')
    if verbose:
        print(f'After dropping duplicates: {len(df)}')

    df = df.dropna(subset=['source', 'pos', 'neg'])
    if verbose:
        print(f'After dropping NAs: {len(df)}')

    # all three columns should be different
    df = df[(df['source'] != df['pos']) & (df['source'] != df['neg']) & (df['pos'] != df['neg'])]
    if verbose:
        print(f'After dropping matches: {len(df)}')

    # all three columns should be non-empty
    df = df[(df['pos'].str.len() > 0) & (df['neg'].str.len() > 0) & (df['source'].str.len() > 0)]
    if verbose:
        print(f'After dropping empty strings: {len(df)}')
    
    return df

def compute_qe_metrics(df, src_col='source', target_col='pos'):
    """
    compute sentence-level quality estimation metrics
    """
    qe_scores = df.apply(lambda row: quality_estimation.sentence_quality_estimation([row[src_col]], [row[target_col]]), axis=1)
    qe_scores = qe_scores.apply(pd.Series)
    for col in qe_scores.columns:
        # convert list to float
        qe_scores[col] = qe_scores[col].apply(lambda x: x[0])

    qe_scores.rename(columns={
        'Compression ratio': f'{target_col}_comp_ratio',
        'Sentence splits': f'{target_col}_sent_splits',
        'Levenshtein similarity': f'{target_col}_lev_sim',
        'Exact copies': f'{target_col}_ex_copies',
        'Additions proportion': f'{target_col}_add_prop',
        'Deletions proportion': f'{target_col}_del_prop',
        'Lexical complexity score': f'{target_col}_lex_comp',
    }, inplace=True)

    df = pd.concat([df, qe_scores], axis=1)

    return df

def compute_bertscore(df, src_col='source', target_col='pos'):
    """
    compute sentence-level BERTScore between source and pos/neg

    Note: the EASSE implentation would load the model for each sentence pair, so we use the implementation from HuggingFace to avoid this
    """
    bertscore = load("bertscore")
    df[f'{target_col}_bertf1'] = df.apply(lambda row: bertscore.compute(predictions=[row[src_col]], references=[row[target_col]], lang="en")['f1'][0], axis=1)

    return df

def compute_fkgl(df, target_col='pos'):
    """
    compute sentence-level FKGL
    """
    df[f'{target_col}_fkgl'] = df.apply(lambda row: fkgl.corpus_fkgl([row[target_col]]), axis=1)
    return df

def drop_outliers(df, target_col='pos', lower_bound=0.01, upper_bound=0.99, verbose=True):
    """
    drop outliers based on FKGL
    """
    if verbose:
        print(f'Initial number of parallel pairs: {len(df)}')

    q_low = df[target_col].quantile(lower_bound)
    q_hi = df[target_col].quantile(upper_bound)

    df = df[(df[target_col] < q_hi) & (df[target_col] > q_low)]
    if verbose:
        print(f'After dropping outliers on {target_col}: {len(df)}')

    return df

def write_to_json_file(df, outfile, question=''):
    """
    Generates a json file following the examples provided in the UniEval repo, e.g.

    {"src": "question: Is this claim consistent with the document? </s> claim: Tunisian referee Slim Jdidi, the man whose performance was close to eclipsing Burkina Faso's amazing fairy-tale step to the African Cup of Nations final, was suspended Thursday by the African Football Confederation. </s> document: (CNN) -- Tunisian referee Slim Jdidi, the man whose performance came close to overshadowing Burkina Faso's astonishing fairytale passage into the final of the Africa Cup of Nations, was suspended by the Confederation of African Football on Thursday. Jdidi's officiating during the Burkinabe's semifinal victory over Ghana has been heavily criticized after he made a number of questionable decisions. Chief amongst those was the 117th-minute sending off of Jonathan Pitroipa for simulation, despite the forward quite clearly being fouled inside the Ghana penalty area. Burkina Faso fairytale ready for one last chapter While Burkina Faso eventually prevailed 3-2 on penalties following a 1-1 draw, all the talk after the game centered on Jdidi's failure to award a legitimate penalty, his decision to disallow a valid goal and the red card he issued to Pitroipa. \"We would have expected a better standard,\" Confederation of African Football (CAF) secretary-general Hicham El Amrani told a media conference. \"There is a meeting tomorrow (Friday) to discuss the Pitroipa incident.\" The Burkinabe are also hopeful that Pitroipa will have his suspension overturned, although that depends on whether Jdidi admits to the error in his match report. \"The organizing committee does not have the power to change the referee's decision unless the referee admits he made a mistake in his official report,\" said El Amrani. Super Eagles' end Ivory Coast's AFCON hopes Burkina Faso's team manager Gualbert Kabore confirmed that the team had appealed against Pitropia's suspension and insists there is a good chance the forward will be available for Sunday's final against Nigeria. Kabore told AFP: \"The Burkina Faso Football Federation wrote an official letter of appeal to CAF (competition organizers). \"We lodged it in the two hours after the match as stipulated by the regulations. \"We think we have a good chance of winning the appeal. \"There were some scandalous decisions, we don't know why, we're asking lots of questions, there are lots of theories. \"Curiously, the players reacted better than us (the team management). They said if the referee is against us that must mean we are the stronger side.\"", "tgt": "Yes"}
    {"src": "question: Is this a fluent paragraph? </s> paragraph: UK police requested a DNA . sample from the girl . It follows latest 'sighting' on New Year's Eve in Queenstown last frequently . is mistaken year Girl for Madeleine, who vanished in 2007 . She volunteered to undergo test to confirm she is not the missing girl .", "tgt": "No"}
    {"src": "question: Is this response consistent with knowledge in the fact? </s> response: the nfl has no written rule against female players, but if they meet the league's eligibility requirements, they would be allowed. </s> fact: the nfl has no written rule against female players; women would in fact be allowed if they met the league's eligibility requirements. an average nfl game only has 11 minutes of live gameplay the average lifetime earnings of an engineer ($5,016,723) is higher than the average lifetime earnings of an nfl player($3,010,000) and an average mlb player ($2,912,000). furthermore, for the elite engineers have higher average career earnings than nba players $13,533,236 vs. $12,027,456. new orleans saints cheerleaders are forbidden from eating in the same restaurant as any nfl player and if they are already dining at a restaurant and an nfl player comes in after, the cheerleaders are required to leave. the nfl uses tracking chips embedded in players shoulder pads to record a players every move on the field. they are also located inside footballs to track: velocity, rotation ,ball speed and location.", "tgt": "Yes"}
    """

    c = 0
    with open(outfile, 'w', encoding='utf8') as outf:
        for i, row in df.iterrows():
            if row['example_type'] == 'pos':
                data = {"src": f"question: {question} </s> text: {row['target']} </s> original: {row['source']}", "tgt": "Yes"}
            elif row['example_type'] == 'neg':
                data = {"src": f"question: {question} </s> text: {row['target']} </s> original: {row['source']}", "tgt": "No"}
            outf.write(json.dumps(data, ensure_ascii=False) + '\n')
            c += 1
    print(f'Wrote {c} examples to {outfile}')
    return

def infer_output_file(output_dir: str, pos_examples: str, neg_examples: str) -> str:
    outfile = Path(output_dir) / f'uni_eval-{Path(pos_examples).stem}-{Path(neg_examples).stem}.json'
    outfile.parent.mkdir(parents=True, exist_ok=True)

    return str(outfile)

if __name__ == '__main__':

    args = parse_args()
    source_sents = list(iter_lines(args.pos_examples, col_num=0))
    pos_examples = list(iter_lines(args.pos_examples, col_num=1))
    neg_examples = list(iter_lines(args.neg_examples, col_num=0))

    assert len(source_sents) == len(pos_examples) == len(neg_examples)

    # infer output file name from input file names
    outfile = infer_output_file(args.output_dir, args.pos_examples, args.neg_examples)
    print(f'Writing to {outfile}...')

    df = pd.DataFrame({'source': source_sents, 'pos': pos_examples, 'neg': neg_examples})
    df = clean_parallel_pairs(df)

    # compute QE metrics between source and pos/neg
    df = compute_qe_metrics(df, src_col='source', target_col='pos')
    df = compute_qe_metrics(df, src_col='source', target_col='neg')

    # compute FKGL for pos/neg/source
    df = compute_fkgl(df, target_col='pos')
    df = compute_fkgl(df, target_col='neg')
    df = compute_fkgl(df, target_col='source')

    # compute BERTScore between source and pos/neg
    df = compute_bertscore(df, src_col='source', target_col='pos')
    df = compute_bertscore(df, src_col='source', target_col='neg')

    df = drop_outliers(df, target_col='neg_lev_sim', lower_bound=0.05, upper_bound=0.95) # anything less than 0.7 seems reasonable upon inspection

    # convert from wide to long (dropping metrics that are not needed for the task)
    df = pd.melt(df, id_vars='source', value_vars=['pos', 'neg'], var_name='example_type', value_name='target')

    if args.num_samples and len(df) > args.num_samples:        
        df = df.sample(n=args.num_samples, random_state=args.seed, replace=False).reset_index(drop=True)
    else:
        print(f'Not sampling because num_samples={args.num_samples} and len(df)={len(df)}')
        df = df.sample(frac=1.0, random_state=args.seed, replace=False).reset_index(drop=True)

    write_to_json_file(df, outfile, question = "Is this text easier to read and understand than the original?")