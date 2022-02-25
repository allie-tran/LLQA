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
from s2s_ft.utils import load_and_cache_examples

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def detokenize(tk_list):
    r_list = []
    for tk in tk_list:
        if tk.startswith('##') and len(r_list) > 0:
            r_list[-1] = r_list[-1] + tk[2:]
        else:
            r_list.append(tk)
    return r_list


_tok_dict = {}

def _is_digit(w):
    for ch in w:
        if not(ch.isdigit() or ch == ','):
            return False
    return True

def fix_tokenization(text):
    input_tokens = text.split()
    output_tokens = []
    i = 0
    prev_dash = False
    while i < len(input_tokens):
        tok = input_tokens[i]
        flag_prev_dash = False
        if tok in _tok_dict.keys():
            output_tokens.append(_tok_dict[tok])
            i += 1
        elif tok == "'" and len(output_tokens) > 0 and output_tokens[-1].endswith("n") and i < len(input_tokens) - 1 and input_tokens[i + 1] == "t":
            output_tokens[-1] = output_tokens[-1][:-1]
            output_tokens.append("n't")
            i += 2
        elif tok == "'" and i < len(input_tokens) - 1 and input_tokens[i + 1] in ("s", "d", "ll"):
            output_tokens.append("'"+input_tokens[i + 1])
            i += 2
        elif tok == "." and i < len(input_tokens) - 2 and input_tokens[i + 1] == "." and input_tokens[i + 2] == ".":
            output_tokens.append("...")
            i += 3
        elif tok == "," and len(output_tokens) > 0 and _is_digit(output_tokens[-1]) and i < len(input_tokens) - 1 and _is_digit(input_tokens[i + 1]):
            # $ 3 , 000 -> $ 3,000
            output_tokens[-1] += ','+input_tokens[i + 1]
            i += 2
        elif tok == "." and len(output_tokens) > 0 and output_tokens[-1].isdigit() and i < len(input_tokens) - 1 and input_tokens[i + 1].isdigit():
            # 3 . 03 -> $ 3.03
            output_tokens[-1] += '.'+input_tokens[i + 1]
            i += 2
        elif tok == "." and len(output_tokens) > 0 and len(output_tokens[-1]) == 1 and output_tokens[-1].isupper() and i < len(input_tokens) - 2 and len(input_tokens[i + 1]) == 1 and input_tokens[i + 1].isupper() and input_tokens[i + 2] == '.':
            # U . N . -> U.N.
            k = i+3
            while k+2 < len(input_tokens):
                if len(input_tokens[k + 1]) == 1 and input_tokens[k + 1].isupper() and input_tokens[k + 2] == '.':
                    k += 2
                else:
                    break
            output_tokens[-1] += ''.join(input_tokens[i:k])
            i += 2
        elif prev_dash and len(output_tokens) > 0 and tok[0] not in string.punctuation:
            output_tokens[-1] += tok
            i += 1
        else:
            output_tokens.append(tok)
            i += 1
        prev_dash = flag_prev_dash
    return " ".join(output_tokens)



class QuestionWordPredictor():
    def __init__(self, model_path, length_penalty, max_length):
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        # Loading tokenizer
        self.tokenizer = UnilmTokenizer.from_pretrained("bert-base-uncased",
                                                        do_lower_case=True, cache_dir=None)
        vocab = self.tokenizer.vocab
        # Loading decoding config
        config_file = os.path.join(model_path, "config.json")
        config = BertConfig.from_json_file(config_file)

        # Preprocessor
        self.bi_uni_pipeline = []
        self.bi_uni_pipeline.append(seq2seq_loader.Preprocess4Seq2seqDecoder(
            list(vocab.keys()), self.tokenizer.convert_tokens_to_ids, 512,
            max_tgt_length=max_length, pos_shift=False,
            source_type_id=config.source_type_id, target_type_id=config.target_type_id,
            cls_token=self.tokenizer.cls_token, sep_token=self.tokenizer.sep_token, pad_token=self.tokenizer.pad_token))

        mask_word_id, eos_word_ids, sos_word_id = self.tokenizer.convert_tokens_to_ids(
            [self.tokenizer.mask_token, self.tokenizer.sep_token, self.tokenizer.sep_token])
        forbid_ignore_set = None

        w_list = ["."]
        forbid_ignore_set = set(self.tokenizer.convert_tokens_to_ids(w_list))

        # Loading model
        logger.info("***** Recover model: %s *****", model_path)
        self.model = BertForSeq2SeqDecoder.from_pretrained(
            model_path, config=config, mask_word_id=mask_word_id, search_beam_size=5,
            length_penalty=length_penalty, eos_id=eos_word_ids, sos_id=sos_word_id,
            forbid_duplicate_ngrams=True, forbid_ignore_set=forbid_ignore_set,
            ngram_size=2, min_len=1, mode='s2s',
            max_position_embeddings=512, pos_shift=False,
        )

        self.model.half()
        self.model.to(self.device)
        self.model.eval()

    def predict_batch(self, examples):
        batch_lines = []
        for example in examples:
            source_tokens = self.tokenizer.tokenize(example)
            batch_lines.append(source_tokens)
        instances = []
        max_a_len = max([len(x) for x in batch_lines])
        for instance in [(x, max_a_len) for x in batch_lines]:
            for proc in self.bi_uni_pipeline:
                instances.append(proc(instance))
        with torch.no_grad():
            batch = seq2seq_loader.batch_list_to_batch_tensors(
                instances)
            batch = [
                t.to(self.device) if t is not None else None for t in batch]
            input_ids, token_type_ids, position_ids, input_mask, mask_qkv, task_idx = batch
            # Run model
            traces = self.model(input_ids, token_type_ids,
                                position_ids, input_mask, task_idx=task_idx, mask_qkv=mask_qkv)
            traces = {k: v.tolist() for k, v in traces.items()}
            output_ids = traces['pred_seq']
            output_sequences = []
            for i in range(len(batch_lines)):
                w_ids = output_ids[i]
                output_buf = self.tokenizer.convert_ids_to_tokens(w_ids)
                output_tokens = []
                for t in output_buf:
                    if t in (self.tokenizer.sep_token, self.tokenizer.pad_token):
                        break
                    output_tokens.append(t)

                output_sequence = ' '.join(
                    detokenize(output_tokens))
                if '\n' in output_sequence:
                    output_sequence = " [X_SEP] ".join(
                        output_sequence.split('\n'))
                output_sequences.append(output_sequence)
            return output_sequences

    def predict(self, text, answer, rest_question=""):
        if rest_question:
            input_text = f"{text} [SEP] {answer} [SEP]  [MASK] {rest_question}"
        else:
            input_text = f"{text} [SEP] {answer}"
        return fix_tokenization(self.predict_batch([input_text])[0].split('?')[0].strip() + '?')


if __name__ == "__main__":
    model = QuestionWordPredictor(
        "/home/tlduyen/LQA/s2s/s2s_ft/full/ckpt-6000/", length_penalty=5, max_length=10)
    print(model.predict(
        "I have to go to Dublin City University to see him", "Dublin City University", "do I have to go to see him ?"))
    
