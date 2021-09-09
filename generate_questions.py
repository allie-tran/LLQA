# Path and logging
import os
import sys
import logging
import warnings
from tqdm import tqdm
import random
warnings.filterwarnings("ignore")
logging.disable(sys.maxsize)

# Other dependencies
from lifelog_tag import SimpleQuestionGenerator, FactMaker, fix_tokenization, ExternalFact
sys.path.insert(1, 'distractor')
from generator import DistractorGenerator

# File formats
from pprint import pprint
from collections import defaultdict
import json

def fix_case(text):
    if text:
        text = text[0].upper() + text[1:].lower()
    return text

class QA:
    """
    An instance of question and answers
    """
    def __init__(self, question, answers):
        qa = {}
        qa["question"] = fix_case(question)
        answers = [fix_case(answer) for answer in answers]
        correct_answer = answers[0]
        random.shuffle(answers)
        qa["answers"] = answers
        qa["answer_idx"] = answers.index(correct_answer)
        self.qa = qa

    def reprJSON(self):
        return self.qa

def make_facts(sents):
    facts = []
    for line in sents.split('\n'):
        if line:
            facts.extend(fact_maker(line))
    return facts

def make_questions(facts):
    questions = []
    for fact, time, truth in facts:
        qu = qu_generator(fact, truth)
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

if __name__ == "__main__":
    # =========================================================================== #
    # IMPORT MODELS FOR QUESTION GENERATION
    # =========================================================================== #
    qu_generator = SimpleQuestionGenerator(
        os.getenv('QUESTION_GEN_PATH'), length_penalty=5, max_length=20)
    fact_maker = FactMaker()
    knowledge = ExternalFact(os.getenv('TFIDF_PATH'))
    distractor = DistractorGenerator(os.getenv('DISTRACTOR_DATA_PATH'))
    
    # Load models
    qu_generator.load()
    fact_maker.load()
    knowledge.load()
    distractor.load()

    sents = sys.argv[-1]
    sents = """We go back to his office and use the laptop. [7pm]
            """
    print("Processing:", sents)
    facts = set(make_facts(sents))
    # =========================================================================== #
    print("-" * 80, "\nMaking questions")
    questions = list(set(make_questions(facts)))
    qa_dict = {}
    for (question, answer), time in questions:
        question = f"{question} [{time}]"
        if question not in qa_dict or len(answer) > len(qa_dict[question]):
            qa_dict[question] = answer
            # print("\t", question, answer)
    questions = list(qa_dict.items())

    if questions:
        # =========================================================================== #
        print("-" * 80, "\nGathering knowledge")
        knowledges = list(knowledge.get_batch_fact(
            [f"{question.split('[')[0] + question.split(']')[-1]} {answer}" for question, answer in questions], 10))
        # =========================================================================== #
        print("-" * 80, "\nGenerating distractors")
        for i, ((question, answer), know) in enumerate(zip(questions, knowledges)):
            if answer in ["yes", "no"]:
                answers = ["yes", "no"] if answer == "yes" else ["no", "yes"]
            else:
                answers = [answer] + distractor(
                        ". ".join([context] + know[:10]), question, answer)
            answers = filter_answers(answers)
            qa = QA(q_count, question, answers, scene['scene']).qa
            q_count += 1
            print(qa)
            

