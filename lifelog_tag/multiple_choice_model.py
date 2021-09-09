"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

import argparse
import json
import logging
import math
import os
import pickle
import random
import string


import numpy as np
import torch
from tqdm import tqdm, trange

import s2s_ft.s2s_loader as seq2seq_loader
from s2s_ft.modeling_decoding import BertConfig, BertForSeq2SeqDecoder
from s2s_ft.tokenization_unilm import UnilmTokenizer
from s2s_ft.modeling import BertForSequenceToSequence
from s2s_ft.config import BertForSeq2SeqConfig
from s2s_ft.configuration_unilm import UnilmConfig
from s2s_ft.utils import load_and_cache_examples, Seq2seqDatasetForBert, batch_list_to_batch_tensors
from torch.utils.data import (DataLoader, SequentialSampler)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class MultipleChoiceAnswerer():
    def __init__(self, model_path):
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        # Loading tokenizer
        self.tokenizer = UnilmTokenizer.from_pretrained("bert-base-uncased",
                                                        do_lower_case=True, cache_dir=None)

        model_config = UnilmConfig.from_pretrained(
            os.path.join(model_path, "config.json"),
            cache_dir=None)
        config = BertForSeq2SeqConfig.from_exist_config(
            config=model_config, label_smoothing=0,
            max_position_embeddings=100)

        self.model = BertForSequenceToSequence.from_pretrained(
            os.path.join(model_path, "pytorch_model.bin"), config=config, model_type='unilm',
            reuse_position_embedding=True,
            cache_dir=None)
        self.model.to(self.device)

    def assess(self, examples):
        features = []
        for example in examples:
            source_tokens = self.tokenizer.tokenize(example["src"])
            target_tokens = self.tokenizer.tokenize(example["tgt"])
            features.append({
                "source_ids": self.tokenizer.convert_tokens_to_ids(source_tokens),
                "target_ids": self.tokenizer.convert_tokens_to_ids(target_tokens),
            })

        train_dataset = Seq2seqDatasetForBert(
            features=features, max_source_len=20,
            max_target_len=10, vocab_size=self.tokenizer.vocab_size,
            cls_id=self.tokenizer.cls_token_id, sep_id=self.tokenizer.sep_token_id, pad_id=self.tokenizer.pad_token_id,
            mask_id=self.tokenizer.mask_token_id, random_prob=0, keep_prob=1,
            offset=0, num_training_instances=len(features),
        )

        train_sampler = SequentialSampler(train_dataset)
        train_dataloader = DataLoader(
            train_dataset, sampler=train_sampler,
            batch_size=1,
            collate_fn=batch_list_to_batch_tensors)

        self.model.eval()
        results = []
        with torch.no_grad():
            for batch in train_dataloader:
                batch = tuple(t.to(self.device) for t in batch)
                inputs = {'source_ids': batch[0],
                          'target_ids': batch[1],
                          'pseudo_ids': batch[2],
                          'num_source_tokens': batch[3],
                          'num_target_tokens': batch[4]}
                loss = self.model(**inputs)
                results.append((np.exp(-loss.item())))
        return results

    def answer(self, question, answers):
        scores = self.assess([{"src": question, "tgt": answer} for answer in answers])
        ans = np.argmax(scores)
        return ans, scores
