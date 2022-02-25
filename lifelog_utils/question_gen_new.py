from spacy.tokens import Doc
import spacy
from spacy.pipeline import merge_entities
from typing import List
from collections import defaultdict
from .spacy_utils import *
import logging
import en_core_web_sm
from pattern3.en import conjugate, PROGRESSIVE
from .qg_pipelines import pipeline

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


def fix_case(text):
    if text:
        text = text[0].upper() + text[1:].lower()
    return text

class SimpleQuestionGenerator:
    def __init__(self):
        self.loaded = False

    def load(self, nlp):
        if self.loaded:
            return
        self.nlp = nlp
        # Question generation
        self.qg = pipeline("question-generation",
                           model="valhalla/t5-base-qg-hl")
        self.loaded = True
        print("Finished loading question generator.")

    @staticmethod
    def _generate_yesno_tobe(text, doc, root, nsubj):
        answer, filter_aux = "yes", [root]
        if root+1 < len(doc) and doc[root+1].dep_ == "neg":
            answer = "no"
            filter_aux.append(root + 1)
        question = " ".join(
            [doc[root].text] + [token.text for token in doc if token.i not in filter_aux and token.text != '.']).strip() + "?"
        return fix_case(question), answer

    @staticmethod
    def _generate_yesno_verb(doc, root, tense, aux):
        answer, filter_aux = "yes", [aux]
        if aux and aux + 1 < len(doc) and doc[aux+1].dep_ == "neg":
            answer = "no"
            filter_aux.append(aux + 1)
        question = " ".join(
            [doc[aux].text if aux else tense] + [token.lemma_ if token.i == root and aux is None else token.text
                                                 for token in doc if token.i not in filter_aux and token.text != '.']).strip() + "?"
        return fix_case(question), answer

    def _generate_how_tobe(self, text, doc, root, nsubj):
        answer, filter_aux = "yes", [root]
        if root+1 < len(doc) and doc[root+1].dep_ == "neg":
            answer = "no"
            filter_aux.append(root + 1)

        # Getting adjs
        adj = None
        for child in doc[root].children:
            if child.dep_ != 'nsubj':
                for r in display_tree([child.i], child, -1):
                    new_filter_aux = filter_aux + r
                    adj = r
                    yield " ".join([doc[i].text for i in adj])
        yield ""

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
                    yield " ".join([doc[i].text for i in advmods])
        yield ""

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
            answer = " ".join([doc[i].text for i in verb[1:]])
            yield answer
        yield ""

    def _generate_subject(self, text, doc, root, nsubj):
        return doc[nsubj].text

    def _generate_object(self, text, doc, root, aux, dobj):
        for r in display_tree([dobj], doc[dobj], -1):
            filter_aux = [aux] + r
            dobj = r
            yield " ".join([doc[i].text for i in dobj])
        yield ""

    def _generate_prep(self, text, doc, root, aux, prep):
        for r in display_tree([prep], doc[prep], prep):
            filter_aux = [aux] + r
            prep_text = r
            yield " ".join([doc[i].text for i in prep_text])
        yield ""

    def get_answers(self, text, truth=True):
        if not self.loaded:
            self.load()
        # text = filter_text(text)
        doc = self.nlp(text)
        root = -1
        answers = []
        yes_no = []
        root = [token.i for token in doc if token.head == token]
        assert root, "Skipping " + text + ": couldn't find root."
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
                        answers.extend(self._generate_object(
                            text, doc, root, root, attr))
                        for attr_child in child.children:
                            if attr_child.dep_ == "prep":
                                answers.extend(self._generate_prep(
                                    text, doc, root, root, attr_child.i))
                    yes_no.append(self._generate_yesno_tobe(
                        text, doc, root, child))
                done = True
        if not done:
            if not nsubj:
                return [], []

            if doc[root].pos_ == "AUX" and doc[root].lemma_ not in ["do", "have"]:
                answers.extend(self._generate_how_tobe(
                    text, doc, root, nsubj))
                yes_no.append(self._generate_yesno_tobe(
                    text, doc, root, nsubj))

            if doc[root].pos_ == "VERB" or doc[root].lemma_ in ["do", "have"]:
                nsubj = None
                for child in doc[root].children:
                    if child.dep_ in ["nsubj", "expl", "nsubjpass"]:
                        nsubj = child
                        if nsubj.lower_ not in ["peter", "there"] and child.pos_ not in ["PRON"]:
                            answers.append(
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
                    root = [
                        token.i for token in doc if token.head == token][0]
                aux = None
                for child in doc[root].children:
                    if child.dep_ in ["aux", "auxpass"]:
                        aux = child.i
                        yes_no.append(
                            self._generate_yesno_tobe(text, doc, aux, nsubj))
                        answers.extend(self._generate_what_action_full(
                            text, doc, root, aux, nsubj))
                if aux:
                    for child in doc[root].children:
                        if child.dep_ == "dobj":
                            dobj = child.i
                            answers.extend(
                                self._generate_object(text, doc, root, aux, child.i))
                            for dobj_child in child.children:
                                if dobj_child.dep_ == "prep":
                                    answers.extend(self._generate_prep(
                                        text, doc, root, aux, dobj_child.i))
                        if child.dep_ == "prep":
                            answers.extend(self._generate_prep(
                                text, doc, root, aux, child.i))
                        if child.dep_ in ["advmod", "acomp"]:
                            answers.extend(
                                self._generate_how(text, doc, root, aux))

        # Filter answers
        answers = [ans for ans in set(
            answers) if ans and "lifelogger" not in ans and len(ans.split()) > 1 and ans not in ["there"]]

        return answers, yes_no

    def yes_no(self, text):
        # text = filter_text(text)
        doc = self.nlp(text)
        root = -1
        yes_no = []
        root = [token.i for token in doc if token.head == token]
        assert root, "Skipping " + text + ": couldn't find root."
        root = root[0]
        nsubj = None
        done = False

        for child in doc[root].children:
            if child.dep_ in ["nsubj", "nsubjpass"]:
                nsubj = child.lower_
            if child.dep_ in ["expl"]:
                for child in doc[root].children:
                    yes_no.append(self._generate_yesno_tobe(
                        text, doc, root, child))
                done = True
        if not done:
            if not nsubj:
                print(text)
                return []

            if doc[root].pos_ == "AUX" and doc[root].lemma_ not in ["do", "have"]:
                yes_no.append(self._generate_yesno_tobe(
                    text, doc, root, nsubj))

            if doc[root].pos_ == "VERB" or doc[root].lemma_ in ["do", "have"]:
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
                    root = [
                        token.i for token in doc if token.head == token][0]
                aux = None
                for child in doc[root].children:
                    if child.dep_ in ["aux", "auxpass"]:
                        aux = child.i
                        yes_no.append(
                            self._generate_yesno_tobe(text, doc, aux, nsubj))
        yes_no = set([q for q, a in yes_no])
        return [(q, "no") for q in yes_no]

    def __call__(self, paragraph):
        sents = []
        answers = []
        yes_no = []
        for sent in paragraph.split('.'):
            sent = sent.strip()
            if sent:
                ans, yn = self.get_answers(sent)
                yes_no.extend([(q, a, sent) for q, a in yn])
                _, more_ans = self.qg._extract_answers(sent)
                for more in more_ans:
                    for a in more:
                        if "lifelogger" in a:
                            continue
                        a = a.replace("<pad>", "").strip()
                        if a not in ans and a not in ["there"]:
                            ans.append(a)
                if ans:
                    sents.append(sent)
                    answers.append([a for a in ans if a in sent])
        output = []
        if sents and answers:
            qg_examples = self.qg._prepare_inputs_for_qg_from_answers_hl(
                sents, answers)
            qg_inputs = [example['source_text'] for example in qg_examples]
            questions = self.qg._generate_questions(qg_inputs)

            sources = []
            for example in qg_examples:
                for sent in sents:
                    if example['answer'] in sent:
                        sources.append(sent)
                        break

            output = [(que, example['answer'], source)
                      for example, que, source in zip(qg_examples, questions, sources)]
        return output, yes_no

if __name__ == "__main__":
    model = SimpleQuestionGenerator()
    model.load()
    print(model('The lifelogger has a white shed in his yard. It is made of wood.'))
