import json
import random
from numpy import source
from .question_gen_new import SimpleQuestionGenerator
from .fact_maker import FactMaker, ExternalFact
from .wh_word_predictor import fix_tokenization
from .distractor_gen import DistractorGenerator
from .spacy_utils import get_nlp
import os
from unicodedata import normalize
import en_core_web_sm


srl_nlp = get_nlp()
qu_generator = SimpleQuestionGenerator()
fact_maker = FactMaker()
knowledge = ExternalFact()
distractor = DistractorGenerator()
nlp = en_core_web_sm.load()

ORIGINAL_LSC = os.getenv('ORIGINAL_LSC')

def load_all():
    qu_generator.load(srl_nlp)
    fact_maker.load(srl_nlp)
    distractor.load()
    knowledge.load()

def make_facts(sents):
    facts = []
    for line in sents.split('\n'):
        if line:
            facts.extend(fact_maker(line))
    return facts


def make_questions(facts):
    questions = []
    paragraph = []
    for fact, time, truth in facts:
        paragraph.append(normalize("NFKD", fact))
    paragraph = ". ".join(paragraph)
    qu = qu_generator(paragraph)
    multiple_questions, binary_questions = qu
    for question in multiple_questions + binary_questions:
        questions.append((question, time))
    return questions


def filter_answers(answers):
    filtered = []
    for answer in answers:
        if answer:
            ans = fix_tokenization(answer)
            filtered.append(ans[0].upper() + ans[1:] + " .")
        else:
            filtered.append(answer)
    return filtered

NUMBERS = ["two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]

def generate_question(sents):
    facts = set(make_facts(sents))
    context = ". ".join([fact[0] for fact in facts])
    questions = list(set(make_questions(facts)))
    qa_dict = {}
    for (question, answer, source_text), time in questions:
        if question not in qa_dict or len(answer) > len(qa_dict[question][0]):
            qa_dict[question] = (answer, source_text)
    questions = qa_dict
    qas = []
    if questions:
        knowledges = list(knowledge.get_batch_fact(
            [(source_text, answer) for answer, source_text in questions.values()], 50))
        for i, ((question, (answer, source_text)), know) in enumerate(zip(questions.items(), knowledges)):
            if answer in ["yes", "no"]:
                answers = ["yes", "no"] if answer == "yes" else ["no", "yes"]
            else:
                if "How many" in question:
                    num = [token.text for token in nlp(answer)][0]
                    wrong_numbers = random.sample([n for n in NUMBERS if n != num], k=3)
                    distractors = [answer.replace(num, wrong) for wrong in wrong_numbers]
                else:
                    doc = " . ".join(know)
                    # print(doc)
                    # print(question)
                    # print(answer)
                    doc += f"</s> {question} </s> {answer} "
                    distractors = distractor.generate(" . ".join(know), question, answer)
                answers = [answer] + distractors
                answers = filter_answers(answers)
            qas.append(
                {"question": question, "answer": "\n".join(answers), "source": source_text})
    return qas

def get_no_question_from_distractor(qas):
    qu_generator.load(srl_nlp)
    no_questions = []
    for qa in qas:
        if "yes" in qa["answer"] and "no" in qa["answer"]:
            pass
        else:
            answers = qa["answer"].split("\n")
            answer = answers[0].lower().strip(' .')
            if answer in qa["source"]:
                for wrong_answer in answers[1:]:
                    wrong_answer = wrong_answer.lower().strip(' .')
                    if answer not in wrong_answer and wrong_answer not in answer:
                        no_qa = qu_generator.yes_no(
                            qa["source"].replace(answer, wrong_answer))
                        for q, _ in no_qa:
                            no_questions.append({"question": q,
                                                "answer": "no\nyes",
                                                "source": qa["source"]})
    return no_questions
