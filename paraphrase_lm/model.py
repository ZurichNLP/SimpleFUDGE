#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pytorch Lightning wrapper for fine-tuning BART.

Adapted from Longformer/LongMBART [1].
Combines functionality from 🤗 fine-tuning script for
summarization [2] and an example of fine-tuning BART on lyrics [3].

[1] https://github.com/a-rios/longmbart/blob/longmbart_hf4/longformer/simplification.py
[2] https://github.com/huggingface/transformers/blob/master/examples/pytorch/summarization/run_summarization_no_trainer.py
[3] https://colab.research.google.com/drive/1Y96pjTQIUvhWlh2GdVRkoyO6MZtLZtpB


BART Special Tokens:
    0: '<s>', (bos_token_id),
    1: '<pad>' (pad_token_id),
    2: '</s>' (decoder_start_token_id and eos_token_id),

"""

import os
import argparse
import random
import numpy as np
import math

import torch
from torch.utils.data import DataLoader, Dataset
from rouge_score import rouge_scorer
import sacrebleu

import pytorch_lightning as pl
from pytorch_lightning.loggers import TestTubeLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.overrides.data_parallel import LightningDistributedDataParallel

import logging
from transformers import (
    BartTokenizer, 
    BartForConditionalGeneration, 
    BartConfig,
    get_linear_schedule_with_warmup, 
    AdamW
)

from transformers.models.bart.modeling_bart import shift_tokens_right
import datasets

from dataset import ParaphraseDataset

# TODO:
# - allow for testing on cpu 
#   raise MisconfigurationException('you selected (distribute_backend = ddp) but did not set Trainer(gpus=?)')
#   pytorch_lightning.utilities.exceptions.MisconfigurationException: you selected (distribute_backend = ddp) but did not set Trainer(gpus=?)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def try_convert_float_to_int(x):
    """
    Parse float argument to type int (required for setting N
    training steps for extremely large data)

    e.g. 
        0.1 -> 10% (as expected)
        10 -> 10 batches
    """
    if x > 1.0:
        return int(x)
    else:
        return float(x)

def freeze_params(model):
    """Set requires_grad=False for each of model.parameters()"""
    for par in model.parameters():
        # print(f'Freezing parameters with shape: {par.shape}')
        par.requires_grad = False

def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=-100):
    """From fairseq"""
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
        count = (~pad_mask).sum()
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
        count = nll_loss.numel()

    nll_loss = nll_loss.sum() / count
    smooth_loss = smooth_loss.sum() / count
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss

def get_eval_scores(gold_strs, generated_strs, vloss=None):
    """
    Computes additional metrics (e.g. BLEU, ROUGE) for a
    batch of generations.
    """
    if vloss is None:
        vloss = torch.zeros(len(gold_strs))
    
    if gold_strs is None and generated_strs is None:
        return {
            'vloss': vloss,
        }
    else:
        scorer = rouge_scorer.RougeScorer(rouge_types=['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=False)
        rouge1 = rouge2 = rougel = rougelsum = 0.0
        # breakpoint()
        for ref, pred in zip(gold_strs, generated_strs):
            score = scorer.score(ref, pred)
            rouge1 += score['rouge1'].fmeasure
            rouge2 += score['rouge2'].fmeasure
            rougel += score['rougeL'].fmeasure
            rougelsum += score['rougeLsum'].fmeasure
        rouge1 /= len(generated_strs)
        rouge2 /= len(generated_strs)
        rougel /= len(generated_strs)
        rougelsum /= len(generated_strs)
        bleu = sacrebleu.corpus_bleu(generated_strs, [gold_strs])

        return {
            'vloss': vloss,
            'rouge1': vloss.new_zeros(1) + rouge1,
            'rouge2': vloss.new_zeros(1) + rouge2,
            'rougeL': vloss.new_zeros(1) + rougel,
            'rougeLsum': vloss.new_zeros(1) + rougelsum,
            'bleu' : vloss.new_zeros(1) + bleu.score,
            # 'decoded' : generated_strs
            }


class LitModel(pl.LightningModule):

    def __init__(self, args):
        super().__init__()
        self.args = args
        
        if self.args.from_pretrained is not None or args.resume_ckpt is not None: ## TODO check if this is true with resume_ckpt
            self._set_config()
            self._load_pretrained()

        self.train_dataloader_object = self.val_dataloader_object = self.test_dataloader_object = None
        self.current_checkpoint = 0
        self.best_checkpoint = None
        self.best_metric = 10000 if self.args.early_stopping_metric == 'vloss' else 0 ## keep track of best dev value of whatever metric is used in early stopping callback
        self.num_not_improved = 0
        self.save_hyperparameters()


    def _load_pretrained(self):
        self.model = BartForConditionalGeneration.from_pretrained(self.args.from_pretrained, config=self.config)
        self.tokenizer = BartTokenizer.from_pretrained(self.args.from_pretrained, use_fast=True)
        
        if self.args.freeze_embeds:
            freeze_params(self.model.model.shared)
            for d in [self.model.model.encoder, self.model.model.decoder]:
                freeze_params(d.embed_positions)
                freeze_params(d.embed_tokens)
        
        if self.args.freeze_encoder:
            freeze_params(self.model.get_encoder())
   
    def _set_config(self):
        self.config = BartConfig.from_pretrained(self.args.from_pretrained)
        self.config.attention_dropout = self.args.attention_dropout
        self.config.dropout = self.args.dropout
        self.config.activation_dropout = self.args.activation_dropout
        self.config.gradient_checkpointing = self.args.grad_ckpt
        # self.config.attention_mode = self.args.attention_mode
        # self.config.attention_window = [self.args.attention_window] * self.config.encoder_layers
    
    def forward(self, input_ids, attention_mask, output_ids):
        """
        Run the forward pass through the model and returns
        the computed loss for current batch. 
        
        The same functionality is used for both the
        train_step and validation_step.
        

        """
        # PREPARE INPUTS FOR BART BEFORE RUNNINF FORWARD PASS

        # Original implementation from LongMBART
        # decoder_input_ids = output_ids[:, :-1].clone() # without eos/last pad
        # labels = output_ids[:, 1:].clone() # without bos

        # Shift the decoder tokens right (but NOT the tgt_ids)
        # https://github.com/huggingface/transformers/blob/7a787c68c6a287ab186f3a099c6496aaee1e8aeb/src/transformers/models/bart/modeling_bart.py#L62
        decoder_input_ids = shift_tokens_right(output_ids, self.tokenizer.pad_token_id, self.model.config.decoder_start_token_id)
        decoder_attention_mask = (decoder_input_ids != self.tokenizer.pad_token_id)
                
        outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                use_cache=False
                )

        lm_logits = outputs[0]
        if self.args.label_smoothing == 0:
            # Same behavior as modeling_bart.py, besides ignoring pad_token_id
            ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
            assert lm_logits.shape[-1] == self.model.config.vocab_size
            loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), output_ids.view(-1))
        else:
            lprobs = torch.nn.functional.log_softmax(lm_logits, dim=-1)
            loss, nll_loss = label_smoothed_nll_loss(
                lprobs, output_ids, self.args.label_smoothing, ignore_index=self.tokenizer.pad_token_id
            )
        return [loss]


    def training_step(self, batch, batch_idx):
        
        output = self.forward(*batch)
        # breakpoint()
        loss = output[0]
        lr = loss.new_zeros(1) + self.trainer.optimizers[0].param_groups[0]['lr']
        # log_info = {
        #     'loss': loss,
        #     'lr': lr,
        #     'input_size': batch[0].numel(),
        #     'output_size': batch[-1].numel(),
        #     'mem': torch.cuda.memory_allocated(loss.device) / 1024 ** 3 if torch.cuda.is_available() else 0
        #     }
        # for k, v in log_info.items():
        #     self.log(k, v, on_step=True, on_epoch=True, prog_bar=True, logger=False)

        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('lr', lr, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        for p in self.model.parameters():
            p.requires_grad = False

        outputs = self.forward(*batch)
        vloss = outputs[0]
        
        if self.args.early_stopping_metric == 'vloss':
            scores = get_eval_scores(None, None, vloss)
        
        # decode validation data
        else: # training time takes much longer
            input_ids, attention_mask, output_ids = batch
            generated_strs = self.generate_text(input_ids, attention_mask)
            gold_strs = self.tokenizer.batch_decode(output_ids.tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
            
            # get scores as dict
            scores = get_eval_scores(gold_strs, generated_strs, vloss)
        
            # write generations to outfile
            outfile = self.args.save_dir + "/" + args.save_prefix + "/_val_out_checkpoint_" + str(self.current_checkpoint)
            with open(outfile, 'a') as f:
                for sample in generated_strs:
                    f.write(sample + "\n")

        # log all scores computed for current batch
        for score_name, score_value in scores.items():
            self.log(score_name, score_value, on_step=False, on_epoch=True, prog_bar=True)

        return scores

    def validation_epoch_end(self, outputs):
        """
        Computes average for all computed scores

        :outputs: [List[Dict]] scores computed for each
            bactch in validation_step()
        """
        for p in self.model.parameters():
            p.requires_grad = True

        names = list(outputs[0].keys())        
        metrics = []
        for name in names:
            metric = torch.stack([x[name] for x in outputs]).mean()
            if self.trainer.use_ddp:
                torch.distributed.all_reduce(metric, op=torch.distributed.ReduceOp.SUM)
                metric /= self.trainer.world_size
            metrics.append(metric)
        # log_info = dict(zip(*[names, metrics]))
        log_info = dict(zip(*[names, [m.item() for m in metrics]]))
        logger.info(f"Validation on checkpoint [{self.current_checkpoint}]: {str(log_info)}")
        
        ## save metric value + number of checkpoint if best
        if self.args.early_stopping_metric == 'vloss':
            if log_info[self.args.early_stopping_metric] < self.best_metric:
                self.best_metric = log_info[self.args.early_stopping_metric]
                self.best_checkpoint = self.current_checkpoint
                logger.info(f"New best checkpoint {self.best_checkpoint}, with {self.best_metric} {self.args.early_stopping_metric}.")
            else:
                logger.info(f"No checkpoint saved: val loss {log_info[self.args.early_stopping_metric]} > {self.best_metric}")
        else:
            if log_info[self.args.early_stopping_metric] > self.best_metric:
                self.best_metric = log_info[self.args.early_stopping_metric]
                self.best_checkpoint = self.current_checkpoint
                logger.info(f"New best checkpoint {self.best_checkpoint}, with {self.best_metric} {self.args.early_stopping_metric}.")
            else:
                logger.info(f"No checkpoint saved: {self.args.early_stopping_metric} {log_info[self.args.early_stopping_metric]} < {self.best_metric}")

        # increment current checkpoint
        self.current_checkpoint += 1
                
    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)

    def test_epoch_end(self, outputs):
        result = self.validation_epoch_end(outputs)
        logger.info(result)
    
    def generate_text(self, input_ids, attention_mask):
        """
        Method to generate text - used in validation_step to
        decode validation data
        """
        # breakpoint()
        generated_ids = self.model.generate(
            input_ids, #inputs["input_ids"],
            attention_mask=attention_mask,#inputs["attention_mask"],
            use_cache=True,
            max_length=self.args.max_output_len,
            num_beams=self.args.beam_size, 
            pad_token_id=self.tokenizer.pad_token_id, decoder_start_token_id=self.tokenizer.pad_token_id,
            early_stopping=True
        )
        
        return self.tokenizer.batch_decode(generated_ids.tolist(), skip_special_tokens=True)

    def configure_optimizers(self):
        """
        Same optimization as BART for CNN/DM.
        Adam for optimizer and lr linearly decayed lr
        scheduler
        (https://github.com/pytorch/fairseq/issues/1364#issuecomment-555176568)
        
        NOTE: User warnings raised can be safely ignored
        - UserWarning: Detected call of
        `lr_scheduler.step()` before `optimizer.step()`
        https://discuss.huggingface.co/t/issues-running-seq2seq-distillation/3075/2?u=tannonk
        - UserWarning: Please also save or load the state of
        the optimizer when saving or loading the scheduler.
        https://github.com/huggingface/transformers/issues/7765#issuecomment-708215509
        """
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        
        # # BUG ReduceLROnPlateau scheduler not functioning
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer, mode=self.lr_mode, factor=self.args.lr_reduce_factor, patience=self.args.lr_reduce_patience)
        # logger.info(f'set reduce LR on plateau schedule with factor={self.args.lr_reduce_factor} and patience={self.args.lr_reduce_patience}.')
        # return {
        #     'optimizer': self.optimizer,
        #     'scheduler': self.scheduler,
        #     'monitor': self.args.early_stopping_metric
        #     }

        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.args.warmup,
            num_training_steps=self.args.max_steps,
        )
        self.scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [self.optimizer], [self.scheduler]

    def _get_dataloader(self, current_dataloader, split_name, is_train):
        if current_dataloader is not None:
            return current_dataloader
       
        dataset = ParaphraseDataset(
            inputs=self.datasets[split_name]['sent1'], 
            labels=self.datasets[split_name]['sent2'],
            name=split_name,
            tokenizer=self.tokenizer,
            max_input_len=self.args.max_input_len,
            max_output_len=self.args.max_output_len,
            )
      
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=is_train) if self.trainer.use_ddp else None

        return DataLoader(
            dataset, 
            batch_size=self.args.batch_size, 
            shuffle=(sampler is None),
            num_workers=self.args.num_workers,
            sampler=sampler,
            collate_fn=ParaphraseDataset.collate_fn
            )

    def train_dataloader(self):
        self.train_dataloader_object = self._get_dataloader(self.train_dataloader_object, 'train', is_train=True)
        return self.train_dataloader_object

    def val_dataloader(self):
        self.val_dataloader_object = self._get_dataloader(self.val_dataloader_object, 'valid', is_train=False)
        return self.val_dataloader_object

    def test_dataloader(self):
        self.test_dataloader_object = self._get_dataloader(self.test_dataloader_object, 'test', is_train=False)
        return self.test_dataloader_object

    def configure_ddp(self, model, device_ids):
        model = LightningDistributedDataParallel(
            model,
            device_ids=device_ids,
            find_unused_parameters=False
        )
        return model


    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument("--tokenizer", type=str, help="Path to the tokenizer directory. If not provided, assumed to be the same as `--from_pretrained`")
        parser.add_argument("--save_dir", type=str, default='simplification', help="Directory to save models.")
        parser.add_argument("--save_prefix", type=str, default='test', help="subfolder in save_dir for this model")
        parser.add_argument("--resume_ckpt", type=str, help="Full path of a checkpoint to resume from")
        parser.add_argument("--from_pretrained", type=str, default=None,  help="Path to a checkpoint to load model weights but not training state")
        parser.add_argument("--num_sanity_val_steps", type=int, default=0,  help="Number of evaluation sanity steps to run before starting the training. Default: 0.")
        
        #data

        parser.add_argument("--train_csv", type=str, default=None,  help="Path to the train csv file.")
        parser.add_argument("--valid_csv", type=str, default=None,  help="Path to the validation csv file.")
        parser.add_argument("--test_csv", type=str, default=None,  help="Path to the test csv file.")
        
        # parser.add_argument("--train_source", type=str, default=None,  help="Path to the source train file.")
        # parser.add_argument("--train_target", type=str, default=None, help="Path to the target train file.")
        # parser.add_argument("--val_source", type=str, default=None, help="Path to the source validation file.")
        # parser.add_argument("--val_target", type=str, default=None, help="Path to the target validation file.")
        # parser.add_argument("--test_source", type=str, default=None, help="Path to the source test file (to evaluate after training is finished).")
        # parser.add_argument("--test_target", type=str, default=None, help="Path to the target test file (to evaluate after training is finished).")
        # parser.add_argument("--src_lang", type=str, default=None, help="Source language tag (optional, for multilingual batches, preprocess text files to include language tags.")
        # parser.add_argument("--tgt_lang", type=str, default=None, help="Target language tag (optional, for multilingual batches, preprocess text files to include language tags.")
        # parser.add_argument("--tags_included", action='store_true', help="Text files already contain special tokens (language tags and </s>. Source:  src_tag seq, Target:  tgt_tag seq. Note: actual source sequence is seq src_tag </s>, will be changed internally after possibly clipping sequence to given max_length.")
        parser.add_argument("--max_output_len", type=int, default=128, help="maximum num of wordpieces/summary. Used for training and testing")
        parser.add_argument("--max_input_len", type=int, default=128, help="maximum num of wordpieces/summary. Used for training and testing")
        parser.add_argument("--wandb", type=str, default=None, help="WandB project name to use if logging fine-tuning with WandB.")
        
        parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
        parser.add_argument("--num_workers", type=int, default=0, help="Number of data loader workers")
        parser.add_argument("--grad_accum", type=int, default=1, help="Number of gradient accumulation steps.")
        parser.add_argument("--gpus", type=int, default=1, help="Number of gpus. 0 for CPU")
        parser.add_argument("--seed", type=int, default=23, help="Seed")
        
        ## model params:
        parser.add_argument("--attention_dropout", type=float, default=0.1, help="attention dropout")
        parser.add_argument("--dropout", type=float, default=0.1, help="dropout")
        parser.add_argument("--activation_dropout", type=float, default=0.0, help="activation_dropout")
        # parser.add_argument("--attention_mode", type=str, default='sliding_chunks', help="Longformer attention mode")
        # parser.add_argument("--attention_window", type=int, default=512, help="Attention window")
        parser.add_argument("--label_smoothing", type=float, default=0.0, required=False)
        # parser.add_argument("--global_attention_indices", type=int, nargs='+', default=[-1], required=False, help="List of indices of positions with global attention for longformer attention. Supports negative indices (-1 == last non-padding token). Default: [-1] == last source token (==lang_id) .")
        parser.add_argument("--freeze_embeds", action="store_true", required=False)
        parser.add_argument("--freeze_encoder", action="store_true", required=False)
        # Optimization params:
        parser.add_argument("--warmup", type=int, default=500, help="Number of warmup steps")
        parser.add_argument("--lr", type=float, default=0.00003, help="Initial learning rate")
        parser.add_argument("--adam_epsilon", type=float, default=1e-8, help="https://pytorch-lightning.readthedocs.io/en/stable/notebooks/lightning_examples/text-transformers.html#Transformer-LightningModule")
        parser.add_argument("--val_check_interval", type=float, default=1.0, help="How often within one training epoch to check the validation set")
        parser.add_argument("--val_percent_check", default=1.00, type=float, help='Percent of validation data used')
        parser.add_argument("--test_percent_check", default=1.00, type=float, help='Percent of test data used')
        parser.add_argument("--train_percent_check", default=1.00, type=float, help='Percent of training data used (for testing) NOTE: not available in pytprch lightning==1.1.6')
        parser.add_argument("--max_epochs", type=int, default=3, help="Maximum number of epochs (will stop training even if patience for early stopping has not been reached).")
        parser.add_argument("--max_steps", type=int, default=0, help="Maximum number of training steps (will stop training even if patience for early stopping has not been reached).")
        
        parser.add_argument("--early_stopping_metric", type=str, default='rougeL', choices=['vloss', 'rouge1', 'rouge2', 'rougeL', 'rougeLsum', 'bleu'], help="Metric to be used for early stopping")
        parser.add_argument("--patience", type=int, default=100, help="Patience for early stopping.")
        parser.add_argument("--lr_reduce_patience", type=int, default=8, help="Patience for LR reduction in Plateau scheduler.")
        parser.add_argument("--lr_reduce_factor", type=float, default=0.5, help="Learning rate reduce factor for Plateau scheduler.")
        parser.add_argument("--weight_decay", type=float, default=0.0, help="Learning rate reduce factor for AdamW")
        parser.add_argument("--disable_checkpointing", action='store_true', help="No logging or checkpointing")
        parser.add_argument("--save_top_k", type=int, default=2, help="Number of best checkpoints to keep. Others will be removed.")
        parser.add_argument("--save_every_n_val_epochs", type=int, default=0, help="Number of validation epochs between checkpoints.")
        parser.add_argument("--save_every_n_train_steps", type=int, default=0, help="Number of training steps between checkpoints.")
        parser.add_argument("--amp_backend", type=str, default='native', help="Number of training steps between checkpoints.")
        parser.add_argument('--grad_ckpt', action='store_true', help='Enable gradient checkpointing to save memory')
        
        ## inference params
        parser.add_argument("--decoded", type=str, default='decoded.out', help="Output file to write decoded sequence to.")
        parser.add_argument("--beam_size", type=int, default=1, help="Beam size for inference when testing/validating. Default: 4.")
        
        #logging params
        parser.add_argument("--progress_bar_refresh_rate", type=int, default=1, help="How often to refresh progress bar (in steps). Value 0 disables progress bar.")
        parser.add_argument("--fp32", action='store_true', help="default is fp16. Use --fp32 to switch to fp32")
        parser.add_argument("--debug", action='store_true', help="debug run")
        parser.add_argument("--print_params", action='store_true', help="Print parameter names and shapes.")
        
        parser.add_argument("--train", action='store_true', help="Whether or not to run training. If not specified, runs testing only")
        

        return parser


def main(args):

    # set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model = LitModel(args)
    
    if args.print_params:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name + ":" + str(param.data.shape))
        exit(0)
      
    # model.datasets = datasets.load_dataset('text', data_files={'train_source': args.train_source, 'train_target': args.train_target, 'val_source': args.val_source, 'val_target': args.val_target, 'test_source': args.test_source, 'test_target': args.test_target })
    
    # expected dataset format is paraNMT_filtered CSV files
    model.datasets = datasets.load_dataset(
        'csv', 
        data_files={
            'train': args.train_csv,
            'valid': args.valid_csv,
            },
        column_names=[
            'temp1','temp2','equality','sent1','sent2',
            'f1_scores','kt_scores','ed_scores','langids'], # columns in csv
        skiprows=1, # csv contains a header
        )

    if not args.test_csv: # assume test set from validation (for debugging)
        model.datasets.update({'test': model.datasets['valid']})
        args.test_csv = True
    
    if args.wandb:
        logger = WandbLogger(project=args.wandb)
    else:
        logger = TestTubeLogger(
            save_dir=args.save_dir,
            name=args.save_prefix,
            version=0  # always use version=0
        )

    print(args)

    lr_monitor_callback = LearningRateMonitor()

    model.lr_mode='max'
    if args.early_stopping_metric == 'vloss':
        model.lr_mode='min'
    early_stop_callback = EarlyStopping(monitor=args.early_stopping_metric, min_delta=0.00, patience=args.patience, verbose=True, mode=model.lr_mode) # metrics: vloss, bleu, rougeL
    
    # breakpoint()
    # Scheduler and math around the number of training
    # steps. Adapted from HF run_finetune.py
    num_update_steps_per_epoch = math.ceil(len(model.datasets['train']) / (args.grad_accum * args.batch_size))
    if args.max_steps == 0: # default
        args.max_steps = args.max_epochs * num_update_steps_per_epoch
    else:
        args.max_epochs = math.ceil(args.max_steps / num_update_steps_per_epoch)

    print(f'Estimated updates per epoch: {num_update_steps_per_epoch}')
    print(f'Inferred max training steps: {args.max_steps} ({args.max_epochs} epochs)')

    custom_checkpoint_path = "ckpt_{{epoch:02d}}_{{{}".format(args.early_stopping_metric)
    custom_checkpoint_path += ':.3f}'

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.save_dir, args.save_prefix),
        filename=custom_checkpoint_path,
        save_top_k=args.save_top_k,
        verbose=True,
        monitor=args.early_stopping_metric,
        mode=model.lr_mode)

    # don't save checkpoints in debug run
    if args.debug:
        args.disable_checkpointing = True

    trainer = pl.Trainer(
        gpus=args.gpus,
        accelerator='ddp' if torch.cuda.is_available() else None,
        track_grad_norm=-1,
        # max_epochs=args.max_epochs if not args.debug else 100,
        max_steps=args.max_steps,
        replace_sampler_ddp=False,
        accumulate_grad_batches=args.grad_accum,
        val_check_interval=try_convert_float_to_int(args.val_check_interval) if not args.debug else 1,
        num_sanity_val_steps=args.num_sanity_val_steps,
        check_val_every_n_epoch=1 if not args.debug else 1,
        limit_val_batches=args.val_percent_check if not args.debug else 2,
        limit_test_batches=args.test_percent_check if not args.debug else 2,
        logger=logger,
        checkpoint_callback=checkpoint_callback if not args.disable_checkpointing else False,
        progress_bar_refresh_rate=args.progress_bar_refresh_rate,
        precision=32 if args.fp32 else 16, amp_level='O2',
        # amp_backend=args.amp_backend,
        resume_from_checkpoint=args.resume_ckpt,
        callbacks=[
            lr_monitor_callback,
            early_stop_callback,
            ]
        )
    
    ## write config + tokenizer to save_dir
    model.model.save_pretrained(args.save_dir + "/" + args.save_prefix)
    model.tokenizer.save_pretrained(args.save_dir + "/" + args.save_prefix)

    if args.train:
        trainer.fit(model)
        print("Training ended. Best checkpoint {} with {} {}.".format(model.best_checkpoint, model.best_metric, args.early_stopping_metric))

        # trainer.test(model)

    # elif args.test_csv:
    #     trainer.test(model)

    # TODO save best checkpoint as pytorch model (not checkpoint)
    # # del model
    # for i, (path, _) in enumerate(trainer.checkpoint_callback.best_k_models.items()):
    #     breakpoint()
    #     model = LitModel.load_from_checkpoint(path)
        # model.model.save_pretrained(f'{i}th_best.pt')
    # print("Decoded outputs written to {}".format(args.translation))


if __name__ == "__main__":
    main_arg_parser = argparse.ArgumentParser(description="Para4Simp")
    parser = LitModel.add_model_specific_args(main_arg_parser)
    args = parser.parse_args()
    main(args)

