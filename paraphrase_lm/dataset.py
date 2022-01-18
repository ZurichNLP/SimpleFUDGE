#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch.utils.data import DataLoader, Dataset

class ParaphraseDataset(Dataset):
    def __init__(self, inputs, labels, name, tokenizer, max_input_len, max_output_len, src_lang='en', tgt_lang='en'):
        self.inputs = inputs
        self.labels = labels
        self.name = name # train, val, test
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        """
        fetches AND encodes a src-tgt pair from the
        dataset given an index
        """
        # breakpoint()
        source = self.inputs[idx]
        encoded_src = self.tokenizer(
            source,
            max_length=self.max_input_len, 
            truncation=True, 
            padding=False,
            return_tensors="pt")
        
        # breakpoint()
        try:
            target = self.labels[idx]
            encoded_tgt = self.tokenizer(target, max_length=self.max_output_len, truncation=True, padding=False, return_tensors="pt")
        except TypeError:
            encoded_tgt = None

        return (
            encoded_src['input_ids'].squeeze(), 
            encoded_src['attention_mask'].squeeze(),
            encoded_tgt['input_ids'].squeeze() if encoded_tgt is not None else None
        )


    @staticmethod
    def collate_fn(batch):
        """
        pads sequences to longest in batch
        """
        pad_token_id = 1 # TODO inherit from loaded tokenizer
        input_ids, attention_masks, output_ids = list(zip(*batch))
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
        attention_masks = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True, padding_value=0)
        # since using the same dataset for train and
        # inference, when doing inference on unseen data,
        # references may not be available.
        if not any(elem is None for elem in output_ids):
            output_ids = torch.nn.utils.rnn.pad_sequence(output_ids, batch_first=True, padding_value=pad_token_id)
        
        return input_ids, attention_masks, output_ids


if __name__ == '__main__':

    import datasets
    from transformers import AutoTokenizer

    # paranmt_filtered data
    train_file = '/srv/scratch6/kew/paraphrase/paraNMT/paranmt_filtered/train_h50.csv'
    dev_file = '/srv/scratch6/kew/paraphrase/paraNMT/paranmt_filtered/dev_h50.csv'
    test_file = None

    tokenizer = AutoTokenizer.from_pretrained('/srv/scratch6/kew/paraphrase/models/paraNMT_filtered/')

    data = datasets.load_dataset(
        'csv', 
        data_files={
            'train': train_file,
            'valid': dev_file,
            },
        column_names=['temp1','temp2','equality','sent1','sent2','f1_scores','kt_scores','ed_scores','langids'],
        skiprows=1, # csv contains a header
        )

    for split in data.keys():
        print(split)
        data_split = ParaphraseDataset(
            inputs=data[split]['sent1'],
            labels=data[split]['sent2'], 
            name=split,
            tokenizer=tokenizer,
            max_input_len=128,
            max_output_len=128
            )
      
        # print(data_split[0])
        # # breakpoint()
        # # sampler = torch.utils.data.distributed.DistributedSampler(data_split, shuffle=(split=='train'))
        # sampler = None

        # print(data_split[0])
        # data_split = DataLoader(
        #     data, batch_size=4, shuffle=(split=='train'),
        #     num_workers=0,
        #     sampler=sampler,
        #     collate_fn=ParaphraseDataset.collate_fn
        #     )
        
        # breakpoint()