from spacy.tokens import Doc
import spacy
from spacy.pipeline import merge_entities
from typing import List
from collections import defaultdict
from .wh_word_predictor import QuestionWordPredictor, fix_tokenization
from .spacy_utils import *
import logging
import text2text as t2t
from pattern3.en import conjugate, PROGRESSIVE

logging.getLogger('allennlp.common.params').disabled = True
logging.getLogger('allennlp.nn.initializers').disabled = True
logging.getLogger(
    'allennlp.modules.token_embedders.embedding').setLevel(logging.INFO)
logging.getLogger('allennlp.common.file_utils').disabled = True
logging.getLogger('transformers.modeling_utils').disabled = True
logging.getLogger('transformers.configuration_utils').disabled = True
logging.getLogger('s2s_ft.s2s_loader').disabled = True
logging.getLogger('s2s_ft.modeling_decoding').disabled = True
logging.getLogger('allennlp.models.archival').disabled = True
logging.getLogger('filelock').disabled = True


def get_wh_word_word(model, text, doc, prep_text, prep, rest_question=""):
    without_prep = " ".join([doc[i].text for i in prep_text if i != prep])
    with_prep = " ".join([doc[i].text for i in prep_text])
    return model.predict(text, with_prep, rest_question), with_prep
    # if without_prep in ["her Macbook"]:
    #     return "What", without_prep
    # return "Where", with_prep


class SimpleQuestionGenerator:
    def __init__(self, model_path, length_penalty=0, max_length=5):
        self.loaded = False
        self.model_path = model_path
        self.length_penalty = length_penalty
        self.max_length = max_length

    def load(self):
        self.qg = t2t.Handler
        self.nlp = spacy.load("en_core_web_sm")
        # Additional pipelines
        self.nlp.add_pipe('srl', after='ner')
        self.nlp.add_pipe("merge_entities")
        self.nlp.add_pipe("merge_noun_chunks")
        self.nlp.add_pipe('merge_srl')
        self.nlp.add_pipe('merge_vp')

        # Wh-word predictor
        self.model = QuestionWordPredictor(
            self.model_path, self.length_penalty, self.max_length)
        self.loaded = True
        print("Finished loading Wh-word predictor.")

    @staticmethod
    def _generate_yesno_tobe(text, doc, root, nsubj):
        answer, filter_aux = "yes", [root]
        if root+1 < len(doc) and doc[root+1].dep_ == "neg":
            answer = "no"
            filter_aux.append(root + 1)
        question = " ".join(
            [doc[root].text] + [token.text for token in doc if token.i not in filter_aux and token.text != '.']).strip() + "?"

        return question, answer

    @staticmethod
    def _generate_yesno_verb(doc, root, tense, aux):
        answer, filter_aux = "yes", [aux]
        if aux and aux + 1 < len(doc) and doc[aux+1].dep_ == "neg":
            answer = "no"
            filter_aux.append(aux + 1)
        question = " ".join(
            [doc[aux].text if aux else tense] + [token.lemma_ if token.i == root and aux is None else token.text
                                                 for token in doc if token.i not in filter_aux and token.text != '.']).strip() + "?"
        return question, answer

    def _generate_how_tobe(self, text, doc, root, nsubj):
        answer, filter_aux = "yes", [root]
        if root+1 < len(doc) and doc[root+1].dep_ == "neg":
            answer = "no"
            filter_aux.append(root + 1)
        # Getting adjs
        adj = None
        for child in doc[root].children:
            if child.dep_ != 'nsubj':
            # if child.dep_ in ["acomp", "advmod", "attr"] or child.pos_ in ["ADJ", "ADV"]:
                for r in display_tree([child.i], child, -1):
                    new_filter_aux = filter_aux + r
                    adj = r
                    rest_question = " ".join(
                        [doc[root].text] + [token.text for token in doc if token.i not in new_filter_aux and token.text != '.']) + "?"
                    question, answer = get_wh_word_word(
                        self.model, text, doc, adj, -1, rest_question)  # How
                    # if "how" not in question:
                        # question = "How " + rest_question
                    yield question, " ".join([nsubj] + [doc[i].text for i in filter_aux] + [answer])
        yield ("", "")

    def _generate_how(self, text, doc, root, aux):
        answer, filter_aux = "yes", [aux]
        if aux+1 < len(doc) and doc[aux+1].dep_ == "neg":
            answer = "no"
            filter_aux.append(aux + 1)

        # Getting advmods
        advmods = []
        for child in doc[root].children:
            if child.dep_ in ["advmod", "acomp"]:
                for r in display_tree([child.i], child, -1):
                    new_filter_aux = filter_aux + r
                    advmods = r
                    rest_question = " ".join(
                        [doc[aux].text] + [token.text for token in doc if token.i not in new_filter_aux and token.text != '.']) + "?"
                    question, answer = get_wh_word_word(
                        self.model, text, doc, advmods, -1, rest_question)
                    yield question, answer
        yield ("", "")

    def _generate_what_action_full(self, text, doc, root, aux, nsubj):
        replace_word = "do" if doc[aux].lemma_ != "be" else "doing"
        dobj = None
        for child in doc[root].children:
            if child.dep_ == "dobj":
                dobj = child.i

        filter_aux = [aux]
        # Getting adjs
        verb = ""
        for r in display_tree([root], doc[root], root):
            filter_aux = [aux] + r
            verb = r
            filter_aux = set(filter_aux).difference({root})
            new_replace_word = replace_word
            if dobj and dobj not in verb:
                new_replace_word += " with"

            # wh_word, _ = get_wh_word_word(self.model, text, doc, verb, -1)  # What
            question = " ".join(["What", doc[aux].text] + [token.text if token.i !=
                                                           root else new_replace_word for token in doc if token.i not in filter_aux and token.text != '.']) + "?"

            if doc[aux].text in ["is", "are"]:
                conj_verb = " ".join([doc[aux].text, conjugate(
                    doc[verb[0]].lemma_, aspect=PROGRESSIVE)])
            else:
                conj_verb = get_conjugation(doc[verb[0]], nsubj)              # 1, 2, 3 or None

            answer = " ".join([nsubj.lower_, conj_verb] + [doc[i].text for i in verb[1:]])
            yield question, answer
        yield ("", "")

    def _generate_what_action(self, text, doc, root, aux):
        replace_word = "do" if doc[aux].lemma_ != "be" else "doing"
        filter_aux = [aux]
        answer = doc[root].text

        for child in doc[root].children:
            if child.dep_ == "dobj":
                replace_word += " with"
                continue
            if child.dep_ == "prt":
                filter_aux += [child.i]
                answer += f" {child.text}"
        question = " ".join(["What", doc[aux].text] + [token.text if token.i !=
                                                       root else replace_word for token in doc if token.i not in filter_aux and token.text != '.']) + "?"
        return question, answer

    def _generate_subject(self, text, doc, root, nsubj):
        # wh_word = get_wh_word_token(doc[nsubj])
        rest_question = " ".join(
            [token.text for token in doc[nsubj+1:] if token.text != '.']) + "?"
        question, answer = get_wh_word_word(
            self.model, text, doc, [nsubj], -1, rest_question)  # How
        return question, answer

    def _generate_object(self, text, doc, root, aux, dobj):
        for r in display_tree([dobj], doc[dobj], -1):
            filter_aux = [aux] + r
            dobj = r
            filter_aux = set(filter_aux).difference({root})

            rest_question = " ".join(
                [doc[aux].text] +
                [token.text for token in doc if token.i not in filter_aux and token.text != '.']) + "?"
            question, answer = get_wh_word_word(
                self.model, text, doc, dobj, -1, rest_question)  # How
            yield question, answer
        yield ("", "")

    def _generate_prep(self, text, doc, root, aux, prep):
        for r in display_tree([prep], doc[prep], prep):
            filter_aux = [aux] + r
            prep_text = r
            rest_question = " ".join([doc[aux].text] +
                                     [token.text for token in doc if token.i not in filter_aux and token.text != '.'] + [doc[prep].text.split()[0]]) + "?"
            question, answer = get_wh_word_word(
                self.model, text, doc, prep_text, prep, rest_question)
            # Remove prep if "when", "where"
            if "when" in question or "where" in question:
                prep_word = question.split()[-1]
                if prep_word not in ["from?", "to?"]:
                    question = " ".join(question.split()[:-1]) + "?"
            yield question, answer
        yield ("", "")

    def __call__(self, text, truth=True):
        if not self.loaded:
            self.load()
        # text = filter_text(text)
        doc = self.nlp(text)
        root = -1
        questions = []
        root = [token.i for token in doc if token.head == token]
        assert root, "can't find root"
        root = root[0]
        nsubj = None
        done = False
        for child in doc[root].children:
            if child.dep_ in ["nsubj", "nsubjpass"]:
                nsubj = child.lower_
            if child.dep_ in ["expl"]:
                for child in doc[root].children:
                    if child.dep_ == "attr":
                        attr = child.i
                        questions.extend(self._generate_object(text, doc, root, root, attr))
                        for attr_child in child.children:
                            if attr_child.dep_ == "prep":
                                questions.extend(self._generate_prep(
                                    text, doc, root, root, attr_child.i))
                    questions.append(self._generate_yesno_tobe(
                        text, doc, root, child))
                done = True

        if not done:
            if not nsubj:
                print("Can't find subject")
                print(text)
                print([token.text for token in doc])
                print([token.dep_ for token in doc])
                return [], []

            if not truth:
                if doc[root].pos_ == "AUX" and doc[root].lemma_ != ["do", "have"]:
                    question, answer = self._generate_yesno_tobe(text, doc, root, nsubj)
                    answer = "yes" if answer == "no" else "no"
                    questions.append((question, answer))

                if doc[root].pos_ == "VERB" or doc[root].lemma_ in ["do", "have"]:
                    for child in doc[root].children:
                        aux = child.i
                        if child.dep_ in ["aux", "auxpass"]:
                            question, answer = self._generate_yesno_tobe(
                                text, doc, aux, nsubj)
                            answer = "yes" if answer == "no" else "no"
                            questions.append((question, answer))
            else:
                if doc[root].pos_ == "AUX" and doc[root].lemma_ not in ["do", "have"]:
                    questions.append(self._generate_yesno_tobe(
                        text, doc, root, nsubj))
                    questions.extend(self._generate_how_tobe(text, doc, root, nsubj))

                if doc[root].pos_ == "VERB" or doc[root].lemma_ in ["do", "have"]:
                    nsubj = None
                    for child in doc[root].children:
                        if child.dep_ in ["nsubj", "expl", "nsubjpass"]:
                            nsubj = child
                            if nsubj.lower_ not in ["peter", "there"] and child.pos_ not in ["PRON"]:
                                questions.append(
                                    self._generate_subject(text, doc, root, child.i))
                    tense = None
                    if doc[root].tag_ == "VBD":
                        tense = "did"
                    elif doc[root].tag_ == "VBP":
                        tense = "do"
                    elif doc[root].tag_ == "VBZ":
                        tense = "does"
                    if tense:
                        new_sent = text.replace(
                            doc[root].text, f"{tense} {doc[root].lemma_}")
                        doc = self.nlp(new_sent)
                        root = [token.i for token in doc if token.head == token][0]
                    aux = None
                    for child in doc[root].children:
                        if child.dep_ in ["aux", "auxpass"]:
                            aux = child.i
                            questions.append(
                                self._generate_yesno_tobe(text, doc, aux, nsubj))
                            # questions.append(self._generate_what_action(text, doc,root,  aux))
                            questions.extend(self._generate_what_action_full(text, doc, root, aux, nsubj))
                    if aux:
                        for child in doc[root].children:
                            if child.dep_ == "dobj":
                                dobj = child.i
                                # questions.extend(
                                #     self._generate_object(text, doc, root, aux, child.i))
                                for dobj_child in child.children:
                                    if dobj_child.dep_ == "prep":
                                        questions.extend(self._generate_prep(text, doc, root, aux, dobj_child.i))
                            if child.dep_ == "prep":
                                questions.extend(self._generate_prep(
                                    text, doc, root, aux, child.i))
                            if child.dep_ in ["advmod", "acomp"]:
                                questions.extend(self._generate_how(text, doc, root, aux))
                    else:
                        print("Wrong POS-tagging in:", text)
                        print([(token.text, token.dep_) for token in doc])
                        # exit()

                for token in doc:
                    if token.pos_ == "SCONJ":
                        try:
                            if token.i == 0 and ', ' in text:
                                answer, rest_question = text.split(', ', 1)
                            elif 'because ' in text:
                                    rest_question, answer = text.split('because ')
                                    answer = "because " + answer
                            else:
                                # print("SCONJ:", token.text, '(', text, ')')
                                questions.append(self.qg(
                                    [f"{text} [SEP] {token.text}"]).question()[0])
                                continue
                            question = self.model.predict(text, answer, rest_question)
                            if question:
                                questions.append((question, answer))
                        except ValueError:
                            print(text)

        lowercases = []
        for question, answer in set(questions):
            if question:
                lowercases.append(
                    (question[0].upper() + question[1:].lower().replace('peter', 'Peter'), answer.replace('peter', 'Peter')))
        # if truth:
        yes_no = [(question, answer) for question,
                    answer in lowercases if answer in ["yes", "no"]]
        if done:
            lowercases = self.qg([f"{text} [SEP] {answer}" for question, answer in lowercases if answer not in ["yes", "no"]]).question()

        lowercases = [(question, answer)
                      for question, answer in lowercases if answer not in ["yes", "no"]]
        return lowercases, yes_no

        def filter_question(question, answer):
            if len(answer.split()) == 1:
                return False
            question = question.lower().split()
            if len(question) == 4 and question[0] == "what" and question[3] == "do?":
                if question[1] in ["do", "does"]:
                    if question[2] in ["he", "peter"]:
                        return False
            return True

        # qa_dict = {}
        # for question, answer in lowercases:
        #     if question not in qa_dict or len(answer) > len(qa_dict[question]):
        #         qa_dict[question] = answer
        # # lowercases = [qa for qa in lowercases if filter_question(*qa)]

        # return list(qa_dict.items())
        return lowercases
