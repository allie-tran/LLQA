#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals
import argparse
import ujson as json

from onmt.utils.logging import init_logger
from onmt.translate.translator import build_translator
import onmt.opts
from eval.eval import eval
import spacy
nlp = spacy.load("en_core_web_sm")
def loads_json(loadpath, loadinfo):
    with open(loadpath, 'r', encoding='utf-8') as fh:
        print(loadinfo)
        dataset = []
        for line in fh:
            example = json.loads(line)
            dataset.append(example)
        print('load json done')
    return dataset

def jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    return len(s1.intersection(s2)) / len(s1.union(s2))

def make_data():
    paragraph = """Cathal wakes up in his bedroom. There was a window in his bedroom.
Cathal goes to the kitchen. He uses his phone in the kitchen.
He opens the fridge. His fridge is full. He takes out food.
The sink pours water into a cup in his hand.
Cathal goes to his living room to eat breakfast with bowl.
He takes his medicine in the kitchen next to the sink. His medicine is in an orange box.
He brushes his teeth in the bathroom. The mirror shows his reflection. His bathroom has pink tiles.
He looks at his reflection in a mirror in his room.
He goes back to the living room. The TV is on. There was an airplane on the TV."""
    ex = {}
    ex["article"] = [token.text for token in nlp(paragraph)]
    ex["answer_text"] = ["a", "cup"]
    ex["distractor"] = ["None"]
    ex["id"] = {"distractor_id": 0, "file_id": 0, "question_id": 0}
    ex["question"] = [token.text for token in nlp("What is he holding next to the sink?")]
    ex["sent"] = [[token.text for token in nlp(sent)] for sent in paragraph.split('.') if sent]
    print(ex)
    return json.dumps(ex)

def main(opt):
    testset = loads_json(opt.data, 'load test file')
    translator = build_translator(opt, report_score=False)
    data = [make_data()]
    translated = translator.translate(data_iter=data, batch_size=opt.batch_size)

    # find first 3 not similar distractors
    hypothesis = {}
    for translation in translated:
        # question_id = str(translation.ex_raw.id['file_id']) + '_' + str(translation.ex_raw.id['question_id'])
        pred1 = translation.pred_sents[0]
        pred2, pred3 = None, None
        for pred in translation.pred_sents[1:]:
            if jaccard_similarity(pred1, pred) < 0.5:
                if pred2 is None:
                    pred2 = pred
                else:
                    if pred3 is None:
                        if jaccard_similarity(pred2, pred) < 0.5:
                            pred3 = pred
            if pred2 is not None and pred3 is not None:
                break

        if pred2 is None:
            pred2 = translation.pred_sents[1]
            if pred3 is None:
                pred3 = translation.pred_sents[2]
        else:
            if pred3 is None:
                pred3 = translation.pred_sents[1]
        print(translation.ex_raw.question)
        # hypothesis[question_id] = [pred1, pred2, pred3]
        print([pred1, pred2, pred3])

    reference = {}
    for sample in testset:
        question_id = str(sample['id']['file_id']) + '_' + str(sample['id']['question_id'])
        if question_id not in reference.keys():
            reference[question_id] = [sample['distractor']]
        else:
            reference[question_id].append(sample['distractor'])
        # print(reference[question_id])

    _ = eval(hypothesis, reference)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='translate.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    onmt.opts.add_md_help_argument(parser)
    onmt.opts.translate_opts(parser)

    opt = parser.parse_args()
    logger = init_logger(opt.log_file)
    main(opt)
