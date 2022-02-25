# Path and logging
from lifelog_utils import generate_question
from pprint import pprint
import os
import sys
import logging
import warnings
import random
import json
warnings.filterwarnings("ignore")
logging.disable(sys.maxsize)

class QA:
    """
    An instance of question and answers
    """

    def __init__(self, question, answers):
        qa = {}
        qa["question"] = fix_case(question.replace('[]', ''))
        answers = [fix_case(answer) for answer in answers]
        correct_answer = answers[0]
        random.shuffle(answers)
        qa["answers"] = answers
        qa["answer_idx"] = answers.index(correct_answer)
        self.qa = qa

    def reprJSON(self):
        return self.qa


if __name__ == "__main__":
    # =========================================================================== #
    # IMPORT MODELS FOR QUESTION GENERATION
    # =========================================================================== #
    sents = """
    The lifelogger is having salmon and eggs on a ceramic plate.
    He is using a fork with his right hand.
    There is a white mug beside the plate with a spoon inside.
    """
    questions = generate_question(sents)
    pprint(questions)
