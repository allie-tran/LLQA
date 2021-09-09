from lifelog_tag import MultipleChoiceAnswerer
import jsonlines
import os
from tqdm import tqdm
import numpy as np
import random

MNT = os.getenv('MNT')


def answer_question(model, qa):
    answers = [qa[f"a{i}"] for i in range(5) if qa[f"a{i}"]]
    random.shuffle(answers)
    if model == "longest answer":
        return answers[np.argmax([len(ans.split()) for ans in answers])]
    elif model == "shortest answer":
        return answers[np.argmin([len(ans.split()) for ans in answers])]
    elif model == "random":
        return random.choice(answers)
    elif model == "similar":
        qu_set = set(qa["q"].split())
        return answers[np.argmax(
            [len(qu_set.intersection(ans.split())) for ans in answers])]
    return ""

acc1 = []
acc2 = []
acc3 = []
acc4 = []
# with jsonlines.open(f"{MNT}/tvqa/full_lifelog/lifelog_test_multiple.jsonl") as reader:
#     for qa in tqdm(reader.iter()):
#         try:
#             answer = answer_question("longest answer", qa)
#             acc1.append(answer == qa[f'a{qa["answer_idx"]}'])
#             answer = answer_question("shortest answer", qa)
#             acc2.append(answer == qa[f'a{qa["answer_idx"]}'])
#             answer = answer_question("random", qa)
#             acc3.append(answer == qa[f'a{qa["answer_idx"]}'])
#             answer = answer_question("similar", qa)
#             acc4.append(answer == qa[f'a{qa["answer_idx"]}'])
#         except ValueError:
#             print(qa)

with jsonlines.open(f"{MNT}/tvqa/full_lifelog/lifelog_test_binary.jsonl") as reader:
    for qa in tqdm(reader.iter()):
        try:
            answer = answer_question("longest answer", qa)
            acc1.append(answer == qa[f'a{qa["answer_idx"]}'])
            answer = answer_question("shortest answer", qa)
            acc2.append(answer == qa[f'a{qa["answer_idx"]}'])
            answer = answer_question("random", qa)
            acc3.append(answer == qa[f'a{qa["answer_idx"]}'])
            answer = answer_question("similar", qa)
            acc4.append(answer == qa[f'a{qa["answer_idx"]}'])
        except ValueError:
            print(qa)

print(len(acc1), round(sum(acc1)/len(acc1), 4))
print(len(acc2), round(sum(acc2)/len(acc2), 4))
print(len(acc3), round(sum(acc3)/len(acc4), 4))
print(len(acc4), round(sum(acc4)/len(acc4), 4))
