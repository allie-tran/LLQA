import spacy
import sys
import os
from .spacy_utils import *
import joblib
import json
import requests
import en_core_web_sm
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

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

    def load(self, nlp):
        if self.loaded:
            return
        self.ori_nlp = en_core_web_sm.load()
        self.ori_nlp.add_pipe('merge_vp')
        self.nlp = nlp
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
        else:
            new_sent = " ".join([replace_amisare(w) for w in ori_doc])
            new_sents.append(new_sent)
        return new_sents

    def __call__(self, line):
        def make_raw_facts(line):
            sents = []
            time = ""
            line = line.replace('(', '[').replace(')', ']')
            if '[' in line:
                time = line.split('[')[-1][:-1].strip(' ]\n')
            for sent in [s.strip() for s in line.split('[')[0].split('.')]:
                if sent:
                    sents.append((f"{sent}", time, True))
            return sents

        sents = make_raw_facts(line)
        info = []
        shared_info = {}
        memo_prev = {}
        memo_next = {}
        new_sents = []
        for text, time, truth in sents:
            sents = self.split(text)
            new_sents.extend(
                [(sent.replace(" which ", " "), time, truth) for sent in sents])
        sents = new_sents
        return sents


def post_mrequest(json_query):
    headers = {"Content-Type": "application/x-ndjson"}
    response = requests.post(
        f"http://localhost:9200/caption/_msearch", headers=headers, data=json_query)
    if response.status_code == 200:
        # stt = "Success"
        response_json = response.json()  # Convert to json as dict formatted
        captions = []
        for res in response_json["responses"]:
            try:
                captions.append([d["_source"]["caption"]
                                  for d in res["hits"]["hits"]])
            except KeyError as e:
                print(res)
                captions.append([])
    else:
        print(f'Response status {response.status_code}')
        captions = []
    return captions

def get_es_query(question_embedding, answer, size=20):
    query = {
        "size": size,
        "_source": {
            "includes": ["caption"]
        },
        "query": {
            "script_score": {
                "query": {
                    "bool":{
                        "must": {"match_all": {}},
                        "must_not": {"match": {
                            "caption": answer
                        }}
                    }
                },
                "script": {
                    "source": "cosineSimilarity(params.queryVector, doc['vector']) + 1.0",
                    "params": {
                        "queryVector": question_embedding.tolist()
                    }
                }
            }
        }
    }
    return query

class ExternalFact:
    def __init__(self):
        self.loaded = False

    def load(self):
        if self.loaded:
            return
        self.model = SentenceTransformer(
            'sentence-transformers/paraphrase-MiniLM-L6-v2')
        # self.unmasker = pipeline('fill-mask', model='bert-base-uncased')
        self.loaded = True
        print("Finished loading sentence embedding model")

    def get_batch_fact(self, sents, max_facts=100):
        try:
            embeddings = self.model.encode([sent for sent, answer in sents if answer not in [
                                           "yes", "no"]], show_progress_bar=False)
            mquery = []
            index = 0
            for (sent, answer) in sents:
                if answer in ["yes", "no"]:
                    continue
                # replacement = answer
                # if answer in sent:
                    # replacement = self.unmasker(sent.replace(answer, "[MASK]"))[
                        # 0]["token_str"]
                mquery.append(json.dumps({}))
                mquery.append(json.dumps(get_es_query(embeddings[index], answer, size=max_facts)))
                index += 1
            results = post_mrequest("\n".join(mquery) + "\n")
            index = 0
            for (sent, answer) in sents:
                if answer in ["yes", "no"]:
                    yield []
                    continue
                else:
                    yield results[index]
                index += 1
        except ValueError:
            return []
