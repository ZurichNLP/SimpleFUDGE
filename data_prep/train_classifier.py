#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script for training Transformer-based classifiers on
simplification level text classification.

Script expects :
    - sentences corresponding to a positive class (e.g.
    simplification level 4 according to Newsela)
    - sentences corresponding to one or more negative classes 

"""

from pathlib import Path
import random
from typing import Dict
import argparse
import numpy as np
import pandas as pd
import evaluate
from datasets import load_dataset, DatasetDict, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

def preprocess_data(file, class_label, tokenize=False):
    with open(file, 'r', encoding='utf8') as f:
        i = 0
        for line in f:
            line = line.strip()
            if line:
                yield {"text": line, "label": class_label}
                i += 1

def write_to_tmp_outfile(data, filepath):
    with open(filepath, 'w', encoding='utf8') as f:
        for item in data:
            f.write(item+'\n')
    return

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def load_data(args) -> Dict:
    dataset = {}
    for split in ['train', 'test', 'valid']:
        # pos class
        p_file = Path(args.data_dir) / f'{args.pos_class}_{split}.txt'
        pdata = list(preprocess_data(p_file, 1))

        # neg class(es)        
        n_files = [Path(args.data_dir) / f'{neg_class}_{split}.txt' for neg_class in args.neg_classes]

        ndata = []
        for n_file in n_files:
            if n_file.exists():
                ndata += list(preprocess_data(n_file, 0))
        random.shuffle(ndata)
        
        # balance out data
        if len(pdata) > len(ndata):
            pdata = pdata[:len(ndata)]
        else:
            ndata = ndata[:len(pdata)]

        data = pdata + ndata
        random.shuffle(data)
        
        # loading directly from list of dicts does not work. `.from_list()` has been deprecated.
        dataset[split] = Dataset.from_pandas(pd.DataFrame.from_records(data))
    
    return dataset
    
# def train(file, epoch=25, lr=0.2):
#     model = fasttext.train_supervised(file, epoch=epoch, lr=lr)
#     return model

# def print_results(N, p, r):
#     print("N\t" + str(N))
#     print("P@{}\t{:.3f}".format(1, p))
#     print("R@{}\t{:.3f}".format(1, r))

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', required=True, type=str, help='data directory')
    parser.add_argument('-o', '--output_dir', required=True, type=str, help='output directory for model and results')
    parser.add_argument('-p', '--pos_class', required=True, type=int, help='positive class')
    parser.add_argument('-n', '--neg_classes', required=False, type=int, nargs='+', help='negative classes. If none given, all other classes are used as negative classes.')
    parser.add_argument('-s', '--seed', required=False, type=int, default=42, help='random seed')
    parser.add_argument('-c', '--clean_up', action='store_true', help='clean up tmp files after training and evaluation')
    parser.add_argument('-m', '--model_name', required=False, type=str, default='distilbert-base-uncased', help='model name')
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = set_args()

    random.seed(args.seed)

    if not args.neg_classes:
        files = Path(args.data_dir).glob('*_train.txt')
        args.neg_classes = [int(f.stem.split('_')[0]) for f in files if int(f.stem.split('_')[0]) != args.pos_class]
        print('Inferred negative classes: ', args.neg_classes)

    # Load data
    dataset = DatasetDict(load_data(args))
    
    # Load model
    model_name = args.model_name
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(set(dataset['train']['label'])))
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    
    # Tokenize datasets
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Define metric
    metric = evaluate.load("accuracy")

    # Train model
    training_args = TrainingArguments(
        output_dir=args.output_dir, 
        evaluation_strategy="steps",
        eval_steps=500,
        num_train_epochs=3,
        max_steps=10000,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=8,
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['valid'],
        compute_metrics=compute_metrics,
    )

    trainer.train()

    trainer.evaluate(
        eval_dataset=tokenized_datasets['test'],
        metric_key_prefix="test",
    )
# # def predict(texts):
# #     inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
# #     outputs = model(**inputs)
# #     logits = outputs.logits
# #     probs = torch.softmax(logits, dim=1)
# #     preds = np.argmax(logits.detach().numpy(), axis=1)
# #     return probs, preds

# # def print_results(probs, preds):
# #     for i, (prob, pred) in enumerate(zip(probs, preds)):
# #         print(f'Example {i+1}:')
# #         print(f'\tProbability of positive class: {prob[1]:.3f}')
# #         print(f'\tPredicted class: {pred}')
# #         print()

