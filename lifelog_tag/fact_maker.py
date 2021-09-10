import spacy
import sys
import os
from .spacy_utils import *
import joblib
import json
try:
    from spacy.lang.en.stop_words import STOP_WORDS
except:
    from spacy.en import STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
import numpy as np
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

sys.path.insert(1, os.getenv('NLI_PATH'))

# from nli_inference import InferenceModel

# inference = InferenceModel()
# inference.load(os.getenv('NLI_PATH'))

def combine_similarities(scores_per_feat, top=10, combine_feat_scores="mul"):
    """
    Get similarities based on multiple independent queries that are then combined using combine_feat_scores
    :param query_feats: Multiple vectorized text queries
    :param para_features: Multiple vectorized text paragraphs that will be scored against the queries
    :param top: Top N facts to keep
    :param combine_feat_scores: The way for combining the multiple scores
    :return: Ranked fact ids with scores List[tuple(id, weight)]
    """
    # scores_per_feat = [pairwise_distances(q_feat, para_features, "cosine").ravel() for q_feat in query_feats]  # this is distance - low is better!!!
    comb_func = np.multiply

    smoothing_val = 0.000001
    max_val = pow((1 + smoothing_val), 2)
    dists = scores_per_feat[0] + smoothing_val
    if len(scores_per_feat) > 1:
        for i in range(1, len(scores_per_feat)):
            dists = comb_func(scores_per_feat[i] + smoothing_val, dists)
    # this is asc (lowers first) ,in case of ties, uses the earlier paragraph
    sorted_ix = np.argsort(dists).tolist()

    max_val = max(np.max(dists), 1)
    return [[i, (max_val - dists[i]) / max_val] for i in sorted_ix][:top]




class FactMaker:
    def __init__(self):
        self.loaded = False

    def load(self):
        self.ori_nlp = spacy.load("en_core_web_sm")
        self.ori_nlp.add_pipe('merge_vp')
        self.nlp = spacy.load("en_core_web_sm")
        # Additional pipelines
        self.nlp.add_pipe('srl', after='ner')
        self.nlp.add_pipe("merge_entities")
        self.nlp.add_pipe("merge_noun_chunks")
        self.nlp.add_pipe('merge_srl')
        self.nlp.add_pipe('merge_vp')
        self.loaded = True
        print("Finished loading fact maker.")

    def split(self, text):
        def replace_amisare(token):
            if token.tag_ in ["VBZ", "VBP", "AUX"] or token.dep_ in ["aux"]:
                if "'s" in token.text:
                    return 'is'
                elif "'re" in token.text:
                    return "are"
            return token.text

        new_sents = []
        ori_doc = self.ori_nlp(text)
        all_tags = list(srl.get_all_tags(ori_doc))
        # print([(w.text, w.tag_) for w in ori_doc])
        # print([(w.text, w.dep_) for w in ori_doc])
        if all_tags:
            for tags, verb in all_tags:
                # print('-' * 80)
                # print(verb)
                # print(tags)

                if verb.dep_ != "ROOT":
                    replace_doc, tags = replace_nsubj(ori_doc, tags)
                else:
                    replace_doc, tags = replace_clause(ori_doc, tags)

                new_sent = " ".join(
                    [replace_amisare(w) for w, tag in zip(replace_doc, tags) if tag != 'O' or (w.head == verb and w.tag_ in ['VBZ', 'VBP'] and w.dep_ != 'conj') or w.dep_ == "expl"])
                if verb.dep_ != "ROOT":
                    root = verb
                    while root.head != root:
                        root = root.head
                        
                    nsubj = None
                    for child in root.children:
                        if child.dep_ in ["nsubj", "attr"]:
                            nsubj = child
                            break
                    if nsubj:
                        new_verb = get_conjugation(verb, nsubj)
                        new_sent = new_sent.replace(verb.text, new_verb)
                if new_sent:
                    new_sents.append(new_sent[0].upper() + new_sent[1:])
                # print('-' * 80)
                # print(verb)
                # print(tags)
                # print([(child.text, child.dep_) for child in verb.children])
                # print(new_sents[-1])
        else:
            new_sent = " ".join([replace_amisare(w) for w in ori_doc])
            new_sents.append(new_sent)
        # input()
        return new_sents

    def __call__(self, line):
        def make_raw_facts(line):
            sents = []
            # time = line.split('[')[-1][:-1]
            # if '-' in time:
            #     time = f"between {' and '.join(time.split('-'))}"
            # else:
            #     time = f"at {time}"
            time = ""
            line = line.replace('(', '[').replace(')', ']')
            if '[' in line:
                time = line.split('[')[-1][:-1].strip(' ]\n')
            for sent in [s.strip() for s in line.split('[')[0].split('.')]:
                if sent:
                    sents.append((f"{sent}", time, True))
            return sents

        # for hypo in inference.infer(line, infer_type="entailment"):
        #     yield hypo, True

        # for hypo in inference.infer(line, infer_type="contradiction"):
        #     yield hypo, False

        sents = make_raw_facts(line)
        info = []
        shared_info = {}
        memo_prev = {}
        memo_next = {}
        new_sents = []
        for text, time, truth in sents:
            sents = self.split(text)
            new_sents.extend([(sent, time, truth) for sent in sents])
        sents = new_sents
        return sents
        # for sent, time in sents:
        #     doc = self.nlp(sent)
        #     main_triple = ["", "", "", defaultdict(lambda: [])]
        #     for token in doc:
        #         # print(token.text, token._.srl, token.head.text)
        #         if token._.srl != 'O':
        #             if token._.srl == "V":
        #                 main_triple[1] = token.text
        #                 for child in token.children:
        #                     if child.dep_ in ['aux', 'auxpass']:
        #                         main_triple[1] = doc[child.i:token.i].text
        #                         break
        #             elif token._.srl == "ARG1":
        #                 main_triple[2] = token.text
        #             elif token._.srl == "ARG0":
        #                 main_triple[0] = token.text
        #             else:
        #                 if token._.srl in ["TMP", "LOC"]:
        #                     if token._.srl not in shared_info :
        #                         shared_info[token._.srl] = token.text
        #                     elif shared_info[token._.srl] != token.text:
        #                         main_triple[3][token._.srl].append(
        #                                                     token.text)
        #                 else:
        #                     if token._.srl in ["ARG4"]:
        #                         memo_prev["LOC"] = token.text.replace("to", "in")
        #                     main_triple[3][token._.srl].append(
        #                         token.text)
        #     main_triple[3] = dict(main_triple[3].items())
        #     memo_prev = {}
        #     info.append(main_triple)

        # for triple, (sent, time) in zip(info, sents):
        #     yield sent.strip(), time, True
        #     # sent = " ".join([sent] + [shared_info[arg] for arg in shared_info if arg in ["TMP", "LOC"]])
        #     all_texts = []
        #     for arg in triple[3]:
        #         if triple[1] in ["go", "goes"] and arg == "ARG4" and len(triple[3][arg]) == 1:
        #             continue
        #         if arg in ["NEG"]:
        #             continue
        #         for text in triple[3][arg]:
        #             all_texts.append(text)
        #     for text in all_texts:
        #         if len(text.split()) > 1:
        #             res = sent.replace(text, "").replace("  ", " ").strip()
        #             yield res, time, True


class ExternalFact:
    class LemmaTokenizer(object):
        def __init__(self):
            self.wnl = WordNetLemmatizer()

        def __call__(self, doc):
            return [self.wnl.lemmatize(t) for t in word_tokenize(doc.lower())]

    def __init__(self, data_path='/home/tlduyen/LQA/mnt/data'):
        self.data_path = data_path

    def load(self):
        self.tfidf = joblib.load(
            f'{self.data_path}/tfidf.joblib')

        self.knowledge_features = joblib.load(
            f'{self.data_path}/knowledge_features.joblib')
        self.facts = []
        with open(f"{self.data_path}/knowledge.json") as f:
            for lid, item in enumerate(f.readlines()):
                json_item = json.loads(item)
                if "surfaceText" in json_item:
                    self.facts.append(json_item["surfaceText"].replace(
                        '[[', '').replace(']]', ''))
                elif "surfaceStart" in json_item:
                    self.facts.append(f"{json_item['surfaceStart']} {json_item['rel']} {json_item['surfaceEnd']}")

    def get_batch_fact(self, sents, max_facts=100):
        try:
            features = self.tfidf.transform(sents)
            feat_similarities = pairwise_distances(
                features, self.knowledge_features, "cosine")
            for feat in feat_similarities:
                ranked = combine_similarities([feat], max_facts,
                                            combine_feat_scores='mul')
                # print('-' * 80)
                facts = []
                for lid, prob in ranked:
                    # print(prob, self.facts[lid])
                    facts.append(self.facts[lid])
                yield facts
        except ValueError:
            return []
