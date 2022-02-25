
from spacy.tokens import Token
from allennlp.predictors.predictor import Predictor
from collections import defaultdict
from spacy.language import Language
from pattern3.en import conjugate
import en_core_web_sm

import inflect
p = inflect.engine()

class SRLComponent(object):
    '''
    A SpaCy pipeline component for SRL
    '''
    name = 'Semantic Role Labeler'

    def __init__(self):
        self.predictor = Predictor.from_path(
            "https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz",
            cuda_device=0,
        )
        Token.set_extension('srl', default="O")

    def __call__(self, doc):
        words = [token.text for token in doc]
        tags = ['O' for _ in words]
        for i, word in enumerate(doc):
            if word.pos_ == "VERB" and word.dep_ != "aux":
                verb = word.text
                verb_labels = [0 for _ in words]
                verb_labels[i] = 1
                instance = self.predictor._dataset_reader.text_to_instance(
                    doc, verb_labels)
                output = self.predictor._model.forward_on_instance(
                    instance)
                new_tags = output['tags']
                for j in range(len(new_tags)):
                    if new_tags[j] != 'O' and tags[j] == 'O':
                        tags[j] = new_tags[j]
        for token, tag in zip(doc, tags):
            token._.set("srl", tag.split('-')[-1])
        return doc

    def get_all_tags(self, doc):
        words = [token.text for token in doc]
        verbs = self.predictor.predict(sentence=" ".join(words))["verbs"]
        for verb in verbs:
            token = [token for token in doc if token.text == verb["verb"]]
            if token:
                token = token[0]
            else:
                token = [token for token in doc if verb["verb"] in token.text.split()][0]
            if token.dep_ != "aux":
                yield verb["tags"], token

        # for i, word in enumerate(doc):
        #     if word.pos_ == "VERB" and word.dep_ != "aux":
        #         verb = word.text
        #         verb_labels = [0 for _ in words]
        #         verb_labels[i] = 1
        #         instance = self.predictor._dataset_reader.text_to_instance(
        #             doc, verb_labels)
        #         output = self.predictor._model.forward_on_instance(
        #             instance)
        #         new_tags = output['tags']
                # yield new_tags, word

srl = SRLComponent()

@Language.component('srl')
def func_srl(doc):
    return srl(doc)


@Language.component('merge_srl')
def merge_srl(doc):
    """Merge semantic roles into a single token.
    doc (Doc): The Doc object.
    RETURNS (Doc): The Doc object with merged noun chunks.
    DOCS: https://spacy.io/api/pipeline-functions#merge_noun_chunks
    """
    if not doc.is_parsed:
        return doc

    with doc.retokenize() as retokenizer:
        spans = []
        attrs = []
        skip = []
        for token in doc:
            if token.i not in skip and token._.srl not in ['O', 'V']:
                srl = token._.srl
                subhead = token
                while subhead.head._.srl == srl and subhead.head != subhead:
                    subhead = subhead.head
                childrens = [subhead]
                expanded = set()
                while True:
                    start_len = len(childrens)
                    for child in childrens:
                        if child.i not in expanded:
                            childrens.extend([c for c in child.children if c._.srl == srl])
                            expanded.add(child.i)
                    end_len = len(childrens)
                    if start_len == end_len:
                        break
                childrens = [w.i for w in childrens]
                start, end = min(childrens), max(childrens) + 1
                attr = {"_": {"srl": srl}}
                spans.append(doc[start:end])
                attrs.append(attr)
                skip.extend(range(start, end))
        try:
            for span, attr in zip(spans, attrs):
                retokenizer.merge(span, attrs=attr)
        except ValueError as e:
            print(spans)
            print(attrs)
            print([(token.text, token._.srl, token.head.text) for token in doc])
            raise e
    return doc


# def get_aux(verb, nsubj):
#     if verb.tag_ == "VBG":
#         return "is"

def get_conjugation(verb, nsubj):
    tags = [child.tag_ for child in verb.children]
    if {'VBZ', 'VBP'}.intersection(tags):
        return conjugate(verb.text, person=3, number="singular")
    if verb.tag_ == 'VBG':
        return ("is " if not p.singular_noun(nsubj.text) else "are ") + conjugate(verb.text, person=3, aspect='progressive')
    return conjugate(verb.text, person=3, number="singular" if not p.singular_noun(nsubj.text) else "plural")

def replace_nsubj(clause, tags):
    wh_ind = [i for i in range(len(clause)) if clause[i].dep_ in [
        'attr', 'nsubj'] and clause[i].tag_ in ['NN', 'WP', 'NNS']]
    if wh_ind:
        wh_ind = wh_ind[0]
        wh_subject = clause[wh_ind]
        if wh_subject.head.dep_ in ['acl', 'relcl']:
            tags = [tags[i] for i in range(len(tags)) if i != wh_ind]
            return [w for w in clause if w != wh_subject], tags
    return clause, tags


def replace_clause(clause, tags):
    relcl_ind = [i for i in range(len(clause)) if clause[i].dep_ == 'relcl']
    if relcl_ind:
        relcl_ind = relcl_ind[0]
        relcl_token = clause[relcl_ind]
        if relcl_token.head.dep_ == 'pobj':
            return clause[:relcl_token.head.i + 1], tags[:relcl_token.head.i + 1]
    return clause, tags


@Language.component('merge_vp')
def merge_vp(doc):
    if not doc.is_parsed:
        return doc
    with doc.retokenize() as retokenizer:
        skip = 0
        for token in doc[:-1]:
            if token.i >= skip and token.pos_ == 'VERB':
                if doc[token.i + 1].dep_ == "prt":
                    retokenizer.merge(
                        doc[token.i:token.i+2], {"LEMMA": f"{token.lemma_} {doc[token.i+1].lemma_}"})
                    skip = token.i+2
    # for token in doc:
    #     print(token.text, token._.srl, token.pos_, token.tag_, token.dep_, token.head.text)
    return doc


phrase_breaker = ['csubj', "acl", "advcl", "appos",
                  "cc", "punct", "ccomp", "xcomp", "amod", "relcl", "dep", "ADP"]

phrase_pause = {"prep"}
child_includes = {
    "conj": [["cc", "punct", "conj"]], "dobj": [["prep"], [""]], "prep": [["prep"]]}
head_includes = {"VERB": [["", "", "prt", 1], ["", "", "advmod", 1]],
                 "NOUN": [["of", "", "prep", 1]],
                 "NUM": [["", "SYM", "nmod", -1]],
                 "ADV": [["", "ADP", "prep", 1]]}


def to_include(head, child):
    if head.pos_ in head_includes:
        for pattern in head_includes[head.pos_]:
            matched = True
            for cond, prop in zip(pattern, (child.text, child.pos_, child.dep_, child.i - head.i)):
                if cond:
                    if isinstance(cond, str) and "~" in cond:
                        if prop == cond:
                            matched = False
                            break
                    elif prop != cond:
                        matched = False
                        break
            if matched:
                return True
    return False


head_breakers = {"NOUN": ["", "~of", "prep"]}

def to_break(head, child):
    if head.pos_ in head_breakers:
        for pattern in head_includes[head.pos_]:
            matched = True
            for cond, prop in zip(pattern, (child.text, child.pos_, child.dep_)):
                if cond:
                    if "~" in cond:
                        if prop == cond:
                            matched = False
                            break
                    elif prop != cond:
                        matched = False
                        break
            if matched:
                return True
    return False


def display_tree(current, token, offset):
    childrens = list(token.children)
    children_dep = [child.dep_ for child in childrens] + \
        [child.pos_ for child in childrens]
    # clauses
    # if {"ccomp", "xcomp"}.intersection(children_dep):
    #     return

    # if phrase_pause.intersection(children_dep):
    #     yield sorted(set(current))

    new_current = []
    for child in childrens:
        if to_include(token, child):
            for res in display_tree(current + [child.i], child, offset):
                new_current.extend(res)  # MULTIPLE?
    if set(new_current).difference(current):
        yield new_current
        current = new_current

    more_current = defaultdict(lambda: current[:])
    deps = []
    for child in childrens:
        if child.dep_ in child_includes:
            for deplist in child_includes[child.dep_]:
                deps.extend(deplist)
    for child in childrens:
        if child.dep_ in deps:
            for res in display_tree(current + [child.i], child, offset):
                more_current[child.dep_].extend(res)  # MULTIPLE?

    got_something = False
    for child in childrens:
        if child.dep_ in phrase_breaker or child.pos_ in phrase_breaker:
            continue
        if to_break(token, child):
            continue
        if child.i > offset:
            currents = []
            if child.dep_ in child_includes:
                for deplist in child_includes[child.dep_]:
                    new_cur = []
                    for dep in deplist:
                        new_cur.extend(more_current[dep])
                    currents.append(new_cur)
            else:
                currents = [current]
            for cur in currents:
                if cur:
                    for res in display_tree(cur + [child.i], child, offset):
                        if set(res).difference(current):
                            yield res
                            got_something = True
    if not got_something:
        yield sorted(set(current))


def filter_text(text):
    replacement = {"in order to": "to",
                   "so that to": "to",
                   "the fact that": "that",
                   "don't": "do not",
                   "doesn't": "does not",
                   "isn't": "is not",
                   "can't": "can not",
                   ".": '',
                   "n't": " not"}
    for r in replacement:
        text = text.replace(r, replacement[r])
    return text

def get_nlp():
    nlp = en_core_web_sm.load()
    # Additional pipelines
    nlp.add_pipe('srl', after='ner')
    nlp.add_pipe("merge_entities")
    nlp.add_pipe("merge_noun_chunks")
    nlp.add_pipe('merge_srl')
    nlp.add_pipe('merge_vp')
    return nlp
