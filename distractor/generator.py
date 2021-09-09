#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals
import argparse
import json
from onmt.utils.logging import init_logger
from onmt.translate.translator import build_translator
import onmt.opts
import spacy

def jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    if s1.issubset(s2) or s2.issubset(s1):
        return 1.0
    return len(s1.intersection(s2)) / len(s1.union(s2))

class DistractorGenerator:
    def __init__(self, model_path='/mnt/DATA/tlduyen/LQA/data/model/jul2_model_step_45000.pt'):
        parser = argparse.ArgumentParser(
            description='translate.py',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        onmt.opts.add_md_help_argument(parser)
        onmt.opts.translate_opts(parser)
        self.opt = parser.parse_args(['-model', model_path, '-share_vocab', '-block_ngram_repeat=1', '-replace_unk',
                                      '-batch_size=1', '-beam_size=50', '-n_best=50', '-gpu=0'])
        self.loaded = False

    def load(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.translator = build_translator(self.opt, report_score=False, out_file="None")
        self.loaded = True

    def _make_data(self, paragraph, question, answer):
        ex = {}
        ex["article"] = [token.text for token in self.nlp(paragraph)]
        ex["answer_text"] = [token.text for token in self.nlp(answer)]
        ex["distractor"] = ["None"]
        ex["id"] = {"distractor_id": 0, "file_id": 0, "question_id": 0}
        ex["question"] = [token.text for token in self.nlp(
            question)]
        ex["sent"] = [[token.text for token in self.nlp(
            sent)] for sent in paragraph.split('.') if sent]
        return json.dumps(ex)

    def __call__(self, paragraph, question, answer):
        data = [self._make_data(paragraph, question, answer)]
        translated = self.translator.translate(data_iter=data, batch_size=self.opt.batch_size)
        hypothesis = {}
        answer = [token.text.replace("'s", "is").replace(
            "cathal", "he") for token in self.nlp(answer)]
        for translation in translated:
            # question_id = str(translation.ex_raw.id['file_id']) + '_' + str(translation.ex_raw.id['question_id'])
            preds = []
            for pred in translation.pred_sents:
                pred = [pred[0].lower()] + [w.replace("'s", "is").replace("cathal", "he") for w in pred[1:]]
                if jaccard_similarity(answer, pred) < 0.5:
                    preds.append(pred)
            pred1 = preds[0]
            pred2, pred3, pred4 = None, None, None
            for pred in preds[1:]:
                if jaccard_similarity(pred1, pred) < 0.5:
                    if pred2 is None:
                        pred2 = pred
                    else:
                        if jaccard_similarity(pred2, pred) < 0.5:
                            if pred3 is None:
                                pred3 = pred
                            # else:
                            #     if jaccard_similarity(pred3, pred) < 0.5:
                            #         pred4 = pred
                if pred2 is not None and pred3 is not None:
                    break

            # if pred2 is None:
            #     pred2 = translation.pred_sents[1]
            #     if pred3 is None:
            #         pred3 = translation.pred_sents[2]
            # else:
            #     if pred3 is None:
            #         pred3 = translation.pred_sents[1]
            return [" ".join(pred).strip('. ') if pred else "" for pred in [pred1, pred2, pred3]]

if __name__ == "__main__":
    distractor_generator = DistractorGenerator()
    distractor_generator.load()
    paragraph = """Cathal wakes up in his bedroom. There was a window in his bedroom.
                    Cathal goes to the kitchen. He uses his phone in the kitchen.
                    He opens the fridge. His fridge is full. He takes out food.
                    The sink pours water into a cup in his hand.
                    Cathal goes to his living room to eat breakfast with bowl.
                    He takes his medicine in the kitchen next to the sink. His medicine is in an orange box.
                    He brushes his teeth in the bathroom. The mirror shows his reflection. His bathroom has pink tiles.
                    He looks at his reflection in a mirror in his room.
                    He goes back to the living room. The TV is on. There was an airplane on the TV."""
    question = "What does he do in the kitchen next to the sink?"
    answer = "take his medicine"
    distractor_generator(paragraph, question, answer)
