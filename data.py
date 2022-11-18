import random
import math
import os
from pathlib import Path
import pickle
from collections import defaultdict, namedtuple
import string
import tokenize
import pandas as pd

os.environ['TOKENIZERS_PARALLELISM'] = 'false' # turn off since we're using multiple threads for loading anyway

from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm
import torch

from util import suppress_stdout
# from poetry_util import is_iambic, count_syllables, get_rhymes, get_rhyme_group
from constants import *

DatasetInfo = namedtuple('DatasetInfo', 
                ['index2word', 'word2index', 'total_words', 'vocab', 'glove_embeddings'])
RhymeInfo = namedtuple('RhymeInfo', 
                ['word2rhyme_group', 'rhyme_group_counts', 'rhyme_groups', 'index2rhyme_group', 'rhyme_group2index', 'total_rhyme_groups'])

SEED = 42
np.random.seed(SEED)

def checker(string):
    # string = string.replace(' ', '')
    string = string.replace('[Ġ▁]', '') # special char for BPE
    return string.strip().lower()


def init_random_embeddings(tokenizer):
    print('Initializing random ebmedding matrix...')
    holder = np.zeros((tokenizer.vocab_size, 300), dtype=np.float32) # ensure float 32 for compatibility during training
    for i in tqdm(range(tokenizer.vocab_size), total=tokenizer.vocab_size):
        holder[i, :] = np.random.rand((300))
    return holder

def map_tokens_to_glove(tokenizer, embedding_file, glove_string, GLOVE_DIM=300):
    """
    Adapted from K2T https://github.com/dapascual/K2T/blob/main/utility_gpt.py
    """
    Path(embedding_file.parent).mkdir(parents=True, exist_ok=True)
    
    if Path(glove_string).exists() and Path(glove_string).is_file():
        print(f'Loading pre-trained embeddings from file {glove_string}')
        glove_encoder = {}
        with open(glove_string, 'r', encoding='utf8') as inf:
            for line in tqdm(inf):
                line = line.strip().split()
                if len(line) != GLOVE_DIM + 1:
                    print(f'Skipping {" ".join(line[:10])}')
                    continue # skip multi-word embeddings which are rare anyway
                glove_encoder[line[0]] = [float(x) for x in line[1:]]
    else:
        print(f'Loading pre-trained embeddings from Gensim ({glove_string})')
        import gensim.downloader as api
        glove_encoder = api.load(glove_string)

    holder = np.zeros((tokenizer.vocab_size, GLOVE_DIM), dtype=np.float32) # ensure float 32 for compatibility during training
    # look up glove representations for each token from the generator model's tokenizer
    null_words = set()

    for i in tqdm(range(tokenizer.vocab_size), total=tokenizer.vocab_size):
        try:
            word = tokenizer.decode([i])
            glove_emb = glove_encoder[checker(word)]
            holder[i, :] = glove_emb
        except:
            word = tokenizer.decode([i])
            null_words.add(word)
            # holder[i, :] = np.zeros((300), dtype=np.float32)
            holder[i, :] = np.random.rand((GLOVE_DIM))

    print(f'Number of token embeddings randomly initialised {len(null_words)}')
    np.save(file=str(embedding_file.with_suffix('')), arr=holder) # save for quicker loading later

    print('Table was generated...')
    return holder

def collate(batch):

    pad_id = batch[0][4]
    inputs = [b[0] for b in batch]
    lengths = torch.LongTensor([b[1] for b in batch])
    max_length = lengths.max()
    for i in range(len(inputs)):
        if len(inputs[i]) < max_length:
            if pad_id == 1:
                inputs[i] = torch.cat([inputs[i], torch.ones(max_length - len(inputs[i])).long()], dim=0)
            else:
                inputs[i] = torch.cat([inputs[i], torch.zeros(max_length - len(inputs[i])).long()], dim=0) # actually 0 is fine as pad since it's masked out
    inputs = torch.stack(inputs, dim=0)
    future_words = torch.LongTensor([b[2] for b in batch]).unsqueeze(0).expand(len(batch), -1).clone() # batch x N=batch
    labels = torch.zeros_like(future_words).long()
    labels = labels.scatter(1, torch.arange(len(batch)).unsqueeze(1), torch.ones(len(batch)).long().unsqueeze(1)).clone()
    log_probs = torch.Tensor([b[3] for b in batch])
    classification_labels = [b[5] for b in batch] # batch
    if type(classification_labels[0]) == list:
        for i in range(len(classification_labels)):
            assert len(classification_labels[i]) == lengths[i]
            if len(classification_labels[i]) < max_length:
                classification_labels[i] = torch.cat([torch.LongTensor(classification_labels[i]), -1 + torch.zeros(max_length - len(classification_labels[i])).long()], dim=0)
            else:
                classification_labels[i] = torch.LongTensor(classification_labels[i])
        classification_labels = torch.stack(classification_labels, dim=0) # batch x seq
    else:
        assert type(classification_labels[0]) == int
        classification_labels = torch.LongTensor(classification_labels) # they're just int labels
    syllables_to_go = torch.LongTensor([b[6] for b in batch])
    future_word_num_syllables = torch.LongTensor([b[7] for b in batch])
    rhyme_group_index = torch.LongTensor([b[8] for b in batch])
    return (inputs, lengths, future_words, log_probs, labels, classification_labels, syllables_to_go, future_word_num_syllables, rhyme_group_index)


def load_rhyme_info(index2word, vocab):
    word2rhyme_group = defaultdict(lambda: UNKNOWN_RHYME_GROUP)
    rhyme_group_counts = defaultdict(lambda: 0)
    rhyme_groups = set()
    for word in index2word:
        try:
            rhyme_group = get_rhyme_group(word)
            word2rhyme_group[word] = rhyme_group
            rhyme_group_counts[rhyme_group] += (vocab[word] if word in vocab else 1) # for rare words not in vocab, just use 1
            rhyme_groups.add(rhyme_group)
        except:
            rhyme_group_counts[UNKNOWN_RHYME_GROUP] += (vocab[word] if word in vocab else 1)
    index2rhyme_group = [UNKNOWN_RHYME_GROUP] + sorted(list(rhyme_groups))
    rhyme_group2index = {s: i for i, s in enumerate(index2rhyme_group)}
    total_rhyme_groups = sum(rhyme_group_counts.values())

    return RhymeInfo(word2rhyme_group=dict(word2rhyme_group), 
                     rhyme_group_counts=dict(rhyme_group_counts), 
                     rhyme_groups=rhyme_groups, 
                     index2rhyme_group=index2rhyme_group, 
                     rhyme_group2index=rhyme_group2index, 
                     total_rhyme_groups=total_rhyme_groups)

def split_line(line):
    line = line.split()
    return [' '.join(line[:i]) for i in range(len(line)) if i > 0]

def split_and_label_for_fudge(line, label, min_length=1, max_length=256):
    line = line.split()
    return [(' '.join(line[:i]), label) for i in range(len(line)) if min_length < i < max_length]


class Dataset:
    def __init__(self, args):
        print('loading data')
        random.seed(args.seed)
        self.batch_size = args.batch_size
        # self.path = Path(args.save_dir) / 'data' / 'dataset_splts.pkl'
        # self.data_dir = args.data_dir
        self.topic = args.task == 'topic'
        self.formality = args.task == 'formality'
        self.iambic = args.task == 'iambic'
        self.rhyme = args.task == 'rhyme'
        self.newline = args.task == 'newline'
        self.simplify = args.task == 'simplify'

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(args.model_path_or_name)
        except:
            tokenizer_name = None 
            if self.formality:
                tokenizer_name = FORMALITY_MODEL_STRING
            elif self.simplify:
                raise RuntimeError
            else:
                tokenizer_name = TOPIC_MODEL_STRING
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        if not self.tokenizer.pad_token:            
            self.tokenizer.add_special_tokens({'pad_token': PAD_TOKEN})
            self.gpt_pad_id = self.tokenizer.encode(PAD_TOKEN, add_special_tokens=False)[0] # actually just the vocab size
        else:
            self.gpt_pad_id = self.tokenizer.pad_token_id

        sentences = []
        self.vocab = defaultdict(lambda: 0)
        if self.formality:
            self.vocab['placeholder'] = 1 # anything so we don't crash
            train, val, test = [], [], []
            for category, label in [('formal', 1), ('informal', 0)]:
                with open(os.path.join(args.data_dir, 'train', category), 'r') as rf:
                    for i, line in enumerate(rf):
                        if len(line) > FORMALITY_MAX_LEN:
                            line = ' '.join(line.strip()[:FORMALITY_MAX_LEN].split()[:-1]) # cutoff words until below max len; chosen so only ~20 examples affected in dataset
                        if i < FORMALITY_VAL_SIZE // 2:
                            val.append((line.strip(), label))
                        else:
                            train.append((line.strip(), label))
                with open(os.path.join(args.data_dir, 'test', category), 'r') as rf:
                    for line in rf:
                        if len(line) > FORMALITY_MAX_LEN:
                            line = ' '.join(line.strip()[:FORMALITY_MAX_LEN].split()[:-1]) # cutoff words until below max len
                        test.append((line.strip(), label))
            self.splits = {}
            self.splits['train'], self.splits['val'], self.splits['test'] = train, val, test

        ####################

        elif self.simplify:

            outpath = Path(args.save_dir) / 'dataset_splts.pkl'
            if outpath.exists():
                with open(outpath, 'rb') as inf:
                    self.splits = pickle.load(inf)
                    print(f'loaded pre-compiled data splits from {outpath}')

            ###########
            # WIKIPEDIA
            ###########
            elif 'wiki' in args.data_dir:

                self.splits = {'train': [], 'val': [], 'test': []}
                
                self.vocab['placeholder'] = 1 # anything so we don't crash
                max_train_lines = 1_000_000
                max_test_val_lines = 5000

                df = pd.read_csv(os.path.join(args.data_dir, 'enwiki_simplewiki.csv'), sep='\t', header=0)
                df['source'].replace(['enwiki', 'simplewiki'], [0, 1], inplace=True)
                df.drop_duplicates(subset=['text'], keep='first', inplace=True) # remove duplicate sents
                df = df[(df['fkgl'] != 0.0)]
                df[(df['fkgl'] > 9.0) & (df['source'] == 1)]

                # remove positive items above a threshold fkgl
                pos_class = df[(df['fkgl'] <= 8.0) & (df['source'] == 1)].reset_index(drop=True)
                # remove negative items below a threshold fkgl
                neg_class = df[(df['fkgl'] >= 10.0) & (df['source'] == 0)].reset_index(drop=True)

                for i, (text, fkgl_score, source) in pos_class.iterrows():
                    
                    if len(self.splits['test']) < (max_test_val_lines // 2):
                        self.splits['test'].extend(split_and_label_for_fudge(text, source))
                    elif len(self.splits['val']) < (max_test_val_lines // 2):
                        self.splits['val'].extend(split_and_label_for_fudge(text, source))
                    elif len(self.splits['train']) < (max_train_lines // 2):
                        self.splits['train'].extend(split_and_label_for_fudge(text, source))
                    else:
                        break
                
                # update max number of train lines if necessary
                # to ensure balanced dataset
                max_train_lines = min(max_train_lines, len(self.splits['train']*2))

                for i, (text, fkgl_score, source) in neg_class.iterrows():
                    if len(self.splits['test']) < max_test_val_lines:
                        self.splits['test'].extend(split_and_label_for_fudge(text, source))
                    elif len(self.splits['val']) < max_test_val_lines:
                        self.splits['val'].extend(split_and_label_for_fudge(text, source))
                    elif len(self.splits['train']) < max_train_lines:
                        self.splits['train'].extend(split_and_label_for_fudge(text, source))
                    else:
                        break
                
                random.Random(SEED).shuffle(self.splits['train'])
                random.Random(SEED).shuffle(self.splits['val'])
                random.Random(SEED).shuffle(self.splits['test'])

                # pickle dataset for later
                with open(outpath, 'wb') as pklf:
                    pickle.dump(self.splits, pklf, pickle.HIGHEST_PROTOCOL)
                print(f'saved data splits in {outpath}')

            #########
            # NEWSELA
            #########
            elif 'newsela' in args.data_dir:
                simp_levels = [0, 1, 2, 3, 4, 5] 
                # simplification levels (aggregated grades in Newsela)
                # 0 = complex (no simplification), 5 = most simplifified

                self.vocab['placeholder'] = 1 # anything so we don't crash
                
                # collect positive samples
                pos_train, pos_val, pos_test = [], [], []
                for split in ['train', 'test', 'valid']:
                    with open(os.path.join(args.data_dir, f'{split}_{args.tgt_level}.txt'), 'r') as rf:
                        for i, line in enumerate(rf):
                            if args.use_line_parts:
                                line_parts = split_line(line.strip()) # this doesn't seem to make a difference
                            else:
                                line_parts = [line.strip()]
                            
                            for lp in line_parts:
                                if split == 'test':
                                    pos_test.append((lp, 1))
                                elif split == 'valid':
                                    pos_val.append((lp, 1))
                                else:
                                    pos_train.append((lp, 1))

                
                # collect all negative samples, i.e. sentences
                # from more complex language levels in Newsela
                neg_train, neg_val, neg_test = [], [], []
                # neg_simp_levels = list(filter(lambda x: x < int(args.tgt_level) simp_levels))
                neg_simp_levels = [0]
                for split in ['train', 'test', 'valid']:
                    for simp_level in neg_simp_levels:
                        with open(os.path.join(args.data_dir, f'{split}_{simp_level}.txt'), 'r') as rf:
                            for i, line in enumerate(rf):
                                if args.use_line_parts:
                                    line_parts = split_line(line.strip()) # this doesn't seem to make a difference
                                else:
                                    line_parts = [line.strip()]

                                for lp in line_parts:
                                    if split == 'test':
                                        neg_test.append((lp, 0))
                                    elif split == 'valid':
                                        neg_val.append((lp, 0))
                                    else:
                                        neg_train.append((lp, 0))

                # shuffle collected negative samples
                random.Random(SEED).shuffle(neg_train)
                random.Random(SEED).shuffle(neg_val)
                random.Random(SEED).shuffle(neg_test)
                self.splits = {}
                self.splits['train'] = pos_train + neg_train[:len(pos_train)]
                self.splits['val'] = pos_val + neg_val[:len(pos_val)]
                self.splits['test'] = pos_test + neg_test[:len(pos_test)]

                random.Random(SEED).shuffle(self.splits['train'])
                random.Random(SEED).shuffle(self.splits['val'])
                random.Random(SEED).shuffle(self.splits['test'])

                # pickle dataset for later
                with open(outpath, 'wb') as pklf:
                    pickle.dump(self.splits, pklf, pickle.HIGHEST_PROTOCOL)
                print(f'saved data splits in {outpath}')

        ####################

            ############
            # APA CAPITO
            ############
            
            elif 'apa_capito' in args.data_dir:
                
                self.vocab['placeholder'] = 1 # anything so we don't crash

                # collect positive samples
                pos_train, pos_val, pos_test = [], [], []
                for split in ['train', 'test', 'dev']:
                    # /srv/scratch6/kew/ats/data/de/apa_capito/article_paragraphs/train_or-A1.de
                    with open(os.path.join(args.data_dir, f'{split}_or-{args.tgt_level}.simpde'), 'r') as rf:
                        for i, line in enumerate(rf):
                            if args.use_line_parts:
                                line_parts = split_line(line.strip()) # this doesn't seem to make a difference
                            else:
                                line_parts = [line.strip()]
                            
                            for lp in line_parts:
                                if len(self.tokenizer.tokenize(lp)) > self.tokenizer.model_max_length:
                                    continue
                                if split == 'test':
                                    pos_test.append((lp, 1))
                                elif split == 'dev':
                                    pos_val.append((lp, 1))
                                else:
                                    pos_train.append((lp, 1))

                
                # collect all negative samples, i.e. sentences
                # from more complex language levels in Newsela
                neg_train, neg_val, neg_test = [], [], []
                # neg_simp_levels = list(filter(lambda x: x < int(args.tgt_level), simp_levels))
                
                for split in ['train', 'test', 'dev']:
                    with open(os.path.join(args.data_dir, f'{split}_or-{args.tgt_level}.de'), 'r') as rf:
                        for i, line in enumerate(rf):
                            if args.use_line_parts:
                                line_parts = split_line(line.strip()) # this doesn't seem to make a difference
                            else:
                                line_parts = [line.strip()]
                            
                            for lp in line_parts:
                                if len(self.tokenizer.tokenize(lp)) > self.tokenizer.model_max_length:
                                    continue
                                if split == 'test':
                                    neg_test.append((lp, 0))
                                elif split == 'dev':
                                    neg_val.append((lp, 0))
                                else:
                                    neg_train.append((lp, 0))

                # shuffle collected negative samples
                random.Random(SEED).shuffle(neg_train)
                random.Random(SEED).shuffle(neg_val)
                random.Random(SEED).shuffle(neg_test)
                self.splits = {}
                self.splits['train'] = pos_train + neg_train[:len(pos_train)]
                self.splits['val'] = pos_val + neg_val[:len(pos_val)]
                self.splits['test'] = pos_test + neg_test[:len(pos_test)]

                random.Random(SEED).shuffle(self.splits['train'])
                random.Random(SEED).shuffle(self.splits['val'])
                random.Random(SEED).shuffle(self.splits['test'])

                # pickle dataset for later
                with open(outpath, 'wb') as pklf:
                    pickle.dump(self.splits, pklf, pickle.HIGHEST_PROTOCOL)
                print(f'saved data splits in {outpath}')

        ####################

        else: # topic / poetry
            for root, _, filenames in os.walk(args.data_dir):
                for fname in filenames:
                    with open(os.path.join(root, fname), 'r') as rf:
                        for line in rf:
                            sentences.append(line.strip())
                            for word in line.strip().split(' '):
                                self.vocab[word] += 1
            random.Random(SEED).shuffle(sentences)
            self.splits = {}
            if args.debug:
                self.splits['val'] = sentences
                self.splits['test'] = sentences
                self.splits['train'] = sentences
            else:
                self.splits['val'] = sentences[:TOPIC_VAL_SIZE]
                self.splits['test'] = sentences[TOPIC_VAL_SIZE:2*TOPIC_VAL_SIZE]
                self.splits['train'] = sentences[2*TOPIC_VAL_SIZE:]

        dataset_info_path = Path(args.save_dir) / 'dataset_info'
        if dataset_info_path.exists():
            print(f'Found exisiting dataset info - loading from file {dataset_info_path}')
            with open(dataset_info_path, 'rb') as rf:
                dataset_info = pickle.load(rf)
            self.vocab, self.total_words, self.index2word, self.word2index, self.glove_embeddings = \
                dataset_info.vocab, dataset_info.total_words, dataset_info.index2word, dataset_info.word2index, dataset_info.glove_embeddings
            self.dataset_info = dataset_info
        
        elif hasattr(args, 'dataset_info') and args.dataset_info is not None:   
            print(f'loading dataset info from file {args.dataset_info}')
            with open(args.dataset_info, 'rb') as rf:
                dataset_info = pickle.load(rf)
            self.vocab, self.total_words, self.index2word, self.word2index, self.glove_embeddings = \
                dataset_info.vocab, dataset_info.total_words, dataset_info.index2word, dataset_info.word2index, dataset_info.glove_embeddings
            self.dataset_info = dataset_info
        
        else: # create dataset info
            # original impl.
            # if args.task != 'simplify':
            #     words_values = list(self.vocab.items())
            #     words_values = sorted(words_values, key=lambda x: x[1], reverse=True)

            print('generating dataset info from scratch')                
            if args.glove is None:
                print('no glove embeddings given')
            
                words_values = list(self.tokenizer.vocab.items())
                words_values = sorted(words_values, key=lambda x: x[1], reverse=False)
                self.vocab = dict(words_values)
                self.total_words = len(self.vocab)
                self.word2index = self.tokenizer.vocab
                self.index2word = {v:k for k, v in self.tokenizer.vocab.items()} # TODO need to remove prefix token?
                # self.vocab = self.tokenizer.vocab
                self.embedding_file = None
                self.glove_embeddings = None
            else:
                # orginal impl.
                if args.task != 'simplify':
                    print('loading glove embeddings')
                    glove_embeddings = {}
                    with open(args.glove, 'r') as rf:
                        for i, line in tqdm(enumerate(rf)):
                            line = line.strip().split()
                            if len(line) != GLOVE_DIM + 1:
                                continue # skip multi-word embeddings which are rare anyway
                            glove_embeddings[line[0]] = [float(x) for x in line[1:]]
                    for word, _ in words_values:
                        if word not in glove_embeddings:
                            del self.vocab[word]

                    self.total_words = sum(self.vocab.values())
                    self.index2word = [PAD_TOKEN] + sorted(list(self.vocab.keys()))
                    self.word2index = {s: i for i, s in enumerate(self.index2word)}
                    self.vocab = dict(self.vocab) # so we can pickle later
                    if glove_embeddings is None:
                        self.glove_embeddings = None
                    else:
                        self.glove_embeddings = torch.stack([torch.zeros(GLOVE_DIM)] + [torch.Tensor(glove_embeddings[word]) for word in self.index2word[1:]], dim=0)

                else: # special handling for simplification
                    # expected glove embedding mapping name should follow form `[tokenizer name]_glove.npy`
                    if Path(args.glove).exists() and Path(args.glove).is_file():
                        embedding_file = Path(args.save_dir) / f"{self.tokenizer.name_or_path.split('/')[-1]}_{Path(args.glove).stem}.npy"
                    else:
                        embedding_file = Path(args.save_dir) / f"{self.tokenizer.name_or_path.split('/')[-1]}_{args.glove}.npy"
                    if not embedding_file.is_file():
                        # create a new mapping if not existing for current tokenizer and glove embs
                        self.glove_embeddings = map_tokens_to_glove(self.tokenizer, embedding_file, args.glove)
                    else:
                        print(f'using existing embedding mapping from {embedding_file}')
                        self.glove_embeddings = np.load(str(embedding_file))
                    self.total_words = len(self.tokenizer.vocab)
                    self.word2index = self.tokenizer.vocab
                    self.index2word = {v:k for k, v in self.tokenizer.vocab.items()} # TODO need to remove prefix token?
                    self.vocab = self.tokenizer.vocab
                    self.embedding_file = str(embedding_file)
                

            self.dataset_info = DatasetInfo(index2word=self.index2word,
                                            word2index=self.word2index,
                                            total_words=self.total_words,
                                            vocab=self.vocab,
                                            glove_embeddings=self.embedding_file)
        
        if self.rhyme:
            if args.rhyme_info is not None:
                print('loading rhyme info from file')
                with open(args.rhyme_info, 'rb') as rf:
                    self.rhyme_info = pickle.load(rf)
            else:
                self.rhyme_info = load_rhyme_info(self.index2word, self.vocab)
            self.word2rhyme_group, self.rhyme_group_counts, self.rhyme_groups, self.index2rhyme_group, self.rhyme_group2index, self.total_rhyme_groups = \
                    defaultdict(lambda: UNKNOWN_RHYME_GROUP, self.rhyme_info.word2rhyme_group), self.rhyme_info.rhyme_group_counts, self.rhyme_info.rhyme_groups, self.rhyme_info.index2rhyme_group, self.rhyme_info.rhyme_group2index, self.rhyme_info.total_rhyme_groups

        print('Done loading data!')
        print('Dataset balance:')
        for split, items in self.splits.items():
            t_cnt = len(self.splits[split])
            s_cnt = sum(1 for i in self.splits[split] if i[1] == 1)
            print(f"complex / simple instances in {split} of size {t_cnt}: {t_cnt - s_cnt} / {s_cnt}")

    def shuffle(self, split, seed=SEED):
        assert split in ['train', 'val', 'test']
        if seed is not None:
            random.seed(seed)
        random.shuffle(self.splits[split])


    def loader(self, split, num_workers=20, indices=None):
        assert split in ['train', 'val', 'test']
        data = self.splits[split] if indices is None else [self.splits[split][i] for i in indices]
        return torch.utils.data.DataLoader(SplitLoader(data, self), batch_size=self.batch_size, pin_memory=True, collate_fn=collate, num_workers=num_workers)


class SplitLoader(torch.utils.data.IterableDataset):
    def __init__(self, data, parent):
        super(SplitLoader).__init__()
        self.data = data
        self.pos = 0
        self.parent = parent


    def __len__(self):
        return len(self.data)


    def __iter__(self):
        return self
    

    def __next__(self):
        increment = 1
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None: # # in a worker process
            increment = worker_info.num_workers
            worker_id = worker_info.id
            if self.pos == 0:
                self.pos = worker_id
        valid = False
        while not valid:
            if self.pos >= len(self):
                raise StopIteration
            if self.parent.topic:
                failed = False
                future_word_num_syllables, rhyme_group_index, syllables_to_go = -1, -1, -1
                raw_sentence, classification_label = self.data[self.pos], -1
                original_sentence = raw_sentence.split()
                sentence = self.parent.tokenizer.encode(raw_sentence, return_tensors='pt')[0]
                length = len(sentence)
                min_sentence_length = MIN_SENTENCE_LENGTH
                if len(sentence) > min_sentence_length: # set to 3. well, everything in data is > 3 for the bag of words task
                    pos_to_split = random.randint(1, length - 1) # for lm, learn all positions at once
                    inp = sentence[:pos_to_split]
                    length = len(inp)
                    num_words_in_input = len(self.parent.tokenizer.decode(inp).split())
                    if not failed and num_words_in_input < len(original_sentence):
                        future_word_position_max = len(original_sentence) - 1
                        future_word_position = random.randint(num_words_in_input-1, future_word_position_max) # allow the last possibly partial word though
                        future_word = original_sentence[future_word_position]
                        unstripped_future_word = future_word
                        future_word = future_word.strip().strip(string.punctuation) # NOTE: we didn't strip punctuation for the topic bag of words paper experiments for our method. it doesn't make much difference, though.
                        if not failed and future_word in self.parent.word2index.keys():
                            word_log_prob = math.log(self.parent.vocab[future_word] / self.parent.total_words) # roughly baseline prob of word under noise model
                            future_word = self.parent.word2index[future_word]
                            pad_id = self.parent.gpt_pad_id
                            example = (inp, length, future_word, word_log_prob, pad_id, classification_label, syllables_to_go, future_word_num_syllables, rhyme_group_index)
                            valid = not failed
            elif self.parent.formality:
                future_word_num_syllables, rhyme_group_index, syllables_to_go = -1, -1, -1
                raw_sentence, classification_label = self.data[self.pos]
                original_sentence = raw_sentence.split()
                sentence = self.parent.tokenizer.encode(raw_sentence, return_tensors='pt')[0]
                length = len(sentence)
                min_sentence_length = MIN_SENTENCE_LENGTH
                if len(sentence) > min_sentence_length: # set to 3. well, everything in data is > 3 for the bag of words task
                    pos_to_split = length # no need to split since we already have the label
                    inp = sentence[:pos_to_split]
                    length = len(inp)
                    num_words_in_input = len(self.parent.tokenizer.decode(inp).split())
                    # only look up to 10 words ahead if we're doing count syllables, since we'll filter out anything more than 10 syllables ahead anyway
                    future_word_position_max = len(original_sentence) - 1
                    future_word_position = 0
                    future_word = 'placeholder'
                    unstripped_future_word = future_word
                    future_word = future_word.strip().strip(string.punctuation) # NOTE: we didn't strip punctuation for the topic bag of words paper experiments for our method. it doesn't make much difference, though.
                    word_log_prob, future_word = 0, 0
                    pad_id = self.parent.gpt_pad_id
                    example = (inp, length, future_word, word_log_prob, pad_id, classification_label, syllables_to_go, future_word_num_syllables, rhyme_group_index)
                    valid = True
            
            #######################

            elif self.parent.simplify:

                future_word_num_syllables, rhyme_group_index, syllables_to_go = -1, -1, -1
                raw_sentence, classification_label = self.data[self.pos]
                original_sentence = raw_sentence.split()
                sentence = self.parent.tokenizer.encode(raw_sentence, return_tensors='pt', add_special_tokens=False)[0]
                length = len(sentence)
                min_sentence_length = MIN_SIMPLIFY_LENGTH
                if len(sentence) > min_sentence_length: # set to 3. well, everything in data is > 3 for the bag of words task
                    pos_to_split = length # no need to split since we already have the label
                    inp = sentence[:pos_to_split]
                    length = len(inp)
                    num_words_in_input = len(self.parent.tokenizer.decode(inp).split())
                    # only look up to 10 words ahead if we're doing count syllables, since we'll filter out anything more than 10 syllables ahead anyway
                    future_word_position_max = len(original_sentence) - 1
                    future_word_position = 0
                    future_word = 'placeholder'
                    unstripped_future_word = future_word
                    future_word = future_word.strip().strip(string.punctuation) # NOTE: we didn't strip punctuation for the topic bag of words paper experiments for our method. it doesn't make much difference, though.
                    word_log_prob, future_word = 0, 0
                    pad_id = self.parent.gpt_pad_id
                    example = (inp, length, future_word, word_log_prob, pad_id, classification_label, syllables_to_go, future_word_num_syllables, rhyme_group_index)
                    valid = True


            elif self.parent.iambic:
                failed = False
                future_word_num_syllables, rhyme_group_index, syllables_to_go = -1, -1, -1
                raw_sentence, classification_label = self.data[self.pos], -1
                original_sentence = raw_sentence.split()
                sentence = self.parent.tokenizer.encode(raw_sentence, return_tensors='pt')[0]
                length = len(sentence)
                min_sentence_length = MIN_SENTENCE_LENGTH
                if len(sentence) > min_sentence_length: # set to 3. well, everything in data is > 3 for the bag of words task
                    pos_to_split = random.randint(0, length - 1)
                    # try to get a subseq of exactly 10 syllables
                    inp = sentence[pos_to_split:]
                    num_syllables = 0
                    checked = False
                    for i in range(1, len(inp)):
                        decoded = self.parent.tokenizer.decode(inp[:i])
                        num_syllables = count_syllables(decoded)
                        if num_syllables > POETRY_LINE_SYLLABLES:
                            inp = inp[:i-1] # might get a few data points where the split is in the middle of a word, but it should be ok for learning. 
                            last_line_length = i-1
                            decoded = self.parent.tokenizer.decode(inp)
                            num_syllables = count_syllables(decoded)
                            checked = True
                            break
                    if not checked or num_syllables != POETRY_LINE_SYLLABLES:
                        failed = True
                    length = len(inp)
                    num_words_in_input = len(self.parent.tokenizer.decode(inp).split())
                    classification_label = [is_iambic(self.parent.tokenizer.decode(inp)) for _ in range(length)] # predict for whole seq including future
                    # only look up to 10 words ahead if we're doing count syllables, since we'll filter out anything more than 10 syllables ahead anyway
                    future_word_position_max = len(original_sentence) - 1
                    future_word_position = 0
                    future_word = 'placeholder'
                    unstripped_future_word = future_word
                    future_word = future_word.strip().strip(string.punctuation) # NOTE: we didn't strip punctuation for the topic bag of words paper experiments for our method. it doesn't make much difference, though.
                    if not failed:
                        word_log_prob, future_word = 0, 0
                        pad_id = self.parent.gpt_pad_id
                        example = (inp, length, future_word, word_log_prob, pad_id, classification_label, syllables_to_go, future_word_num_syllables, rhyme_group_index)
                        valid = not failed
            elif self.parent.rhyme:
                failed = False
                future_word_num_syllables, rhyme_group_index = -1, -1
                raw_sentence, classification_label = self.data[self.pos], -1
                original_sentence = raw_sentence.split()
                sentence = self.parent.tokenizer.encode(raw_sentence, return_tensors='pt')[0]
                length = len(sentence)
                min_sentence_length = MIN_SENTENCE_LENGTH
                if len(sentence) > min_sentence_length: # set to 3. well, everything in data is > 3 for the bag of words task
                    pos_to_split = random.randint(1, length - 1) # for lm, learn all positions at once
                    inp = sentence[:pos_to_split]
                    length = len(inp)
                    num_words_in_input = len(self.parent.tokenizer.decode(inp).split())
                    if not failed and num_words_in_input < len(original_sentence):
                        # only look up to 10 words ahead if we're doing count syllables, since we'll filter out anything more than 10 syllables ahead anyway
                        future_word_position_max = min(len(original_sentence) - 1, num_words_in_input + MAX_COUNT_SYLLABLE_DIST)
                        future_word_position = random.randint(num_words_in_input-1, future_word_position_max) # allow the last possibly partial word though
                        future_word = original_sentence[future_word_position]
                        unstripped_future_word = future_word
                        future_word = future_word.strip().strip(string.punctuation) # NOTE: we didn't strip punctuation for the topic bag of words paper experiments for our method. it doesn't make much difference, though.
                                                        
                        words_in_between = original_sentence[num_words_in_input-1:future_word_position+1]
                        syllables_to_go = count_syllables(' '.join(words_in_between))
                        if syllables_to_go > MAX_COUNT_SYLLABLE_DIST:
                            failed = True
                        future_word_num_syllables = count_syllables(future_word)
                        rhyme_group = self.parent.word2rhyme_group[future_word]
                        rhyme_group_index = self.parent.rhyme_group2index[rhyme_group]
                        # truncate context a bit since we're just doing couplets. random length from 1 to max desired length for this purpose. 
                        desired_length = random.randint(1, MAX_COUNT_SYLLABLE_INPUT_LENGTH)
                        inp = inp[-desired_length:]
                        length = len(inp)

                        if not failed and future_word in self.parent.word2index.keys():
                            word_log_prob = math.log(self.parent.rhyme_group_counts[rhyme_group] / self.parent.total_rhyme_groups)
                            future_word = rhyme_group_index # future conditioning is just the rhyme group in this case
                            pad_id = self.parent.gpt_pad_id
                            example = (inp, length, future_word, word_log_prob, pad_id, classification_label, syllables_to_go, future_word_num_syllables, rhyme_group_index)
                            valid = not failed
            elif self.parent.newline:
                failed = False
                future_word_num_syllables, rhyme_group_index = -1, -1
                raw_sentence, classification_label = self.data[self.pos], -1
                original_sentence = raw_sentence.split()
                sentence = self.parent.tokenizer.encode(raw_sentence, return_tensors='pt')[0]
                length = len(sentence)
                min_sentence_length = MIN_SENTENCE_LENGTH
                if len(sentence) > min_sentence_length: # set to 3. well, everything in data is > 3 for the bag of words task
                    pos_to_split = random.randint(1, length - 1) # for lm, learn all positions at once
                    inp = sentence[:pos_to_split]
                    while pos_to_split < len(sentence):
                        if len(self.parent.tokenizer.decode(inp).split()) == len(self.parent.tokenizer.decode(sentence[:pos_to_split + 1]).split()):
                            pos_to_split += 1
                            inp = sentence[:pos_to_split]
                        else:
                            break
                    length = len(inp)
                    num_words_in_input = len(self.parent.tokenizer.decode(inp).split())
                    if not failed and num_words_in_input < len(original_sentence):
                        # only look up to 10 words ahead if we're doing count syllables, since we'll filter out anything more than 10 syllables ahead anyway
                        future_word_position_max = len(original_sentence) - 1
                        future_word_position = random.randint(num_words_in_input-1, future_word_position_max) # allow the last possibly partial word though
                        future_word = original_sentence[future_word_position]
                        unstripped_future_word = future_word
                        future_word = future_word.strip().strip(string.punctuation) # NOTE: we didn't strip punctuation for the topic bag of words paper experiments for our method. it doesn't make much difference, though.
                                                        
                        # future_word = original_sentence[-1] # useful for debugging
                        words_in_between = original_sentence[num_words_in_input-1:future_word_position+1]
                        syllables_to_go = count_syllables(' '.join(words_in_between))
                        if syllables_to_go > MAX_COUNT_SYLLABLE_DIST:
                            failed = True
                        # truncate context a bit since we're just doing couplets. random length from 1 to max desired length for this purpose. 
                        desired_length = random.randint(1, MAX_COUNT_SYLLABLE_INPUT_LENGTH)
                        # desired_length = 10 # useful for debugging
                        inp = inp[-desired_length:]
                        length = len(inp)
                        true_label = 1 if unstripped_future_word.strip()[-1] in PHRASE_ENDS else 0 # common ways to end a phrase
                        classification_label = [-1 for _ in range(length)]
                        classification_label[-1] = true_label # only learn at the last position
                        if not failed and future_word in self.parent.word2index.keys():
                            word_log_prob = math.log(self.parent.vocab[future_word] / self.parent.total_words) # roughly baseline prob of word under noise model
                            future_word = self.parent.word2index[future_word]
                            pad_id = self.parent.gpt_pad_id
                            example = (inp, length, future_word, word_log_prob, pad_id, classification_label, syllables_to_go, future_word_num_syllables, rhyme_group_index)
                            valid = not failed
            else:
                raise NotImplementedError

            self.pos += increment
        return example
