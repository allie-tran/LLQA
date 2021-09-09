from lifelog_tag import MultipleChoiceAnswerer
import jsonlines
import os
from tqdm import tqdm

model_1 = MultipleChoiceAnswerer(os.getenv('MULTIPLE_ANSWERER_PATH'))
model_2 = MultipleChoiceAnswerer(os.getenv('MULTIPLE_ANSWERER_PATH_KNOWLEDGE'))
MNT = os.getenv('MNT')

def answer_question(model, qa):
    answers = [qa[f"a{i}"] for i in range(5) if qa[f"a{i}"]]
    answer, scores = model.answer(qa["q"], answers)
    return answers[answer]

acc1 = []
acc2 = []
with jsonlines.open(f"{MNT}/tvqa/full_lifelog/lifelog_test_multiple.jsonl") as reader:
    for qa in tqdm(reader.iter()):
        try:
            answer = answer_question(model_1, qa)
            acc1.append(answer == qa[f'a{qa["answer_idx"]}'])
            answer = answer_question(model_2, qa)
            acc2.append(answer == qa[f'a{qa["answer_idx"]}'])
        except ValueError:
            print(qa)

# with jsonlines.open(f"{MNT}/tvqa/full_lifelog/lifelog_test_binary.jsonl") as reader:
#     for qa in tqdm(reader.iter()):
#         try:
#             answer = answer_question(model_1, qa)
#             acc1.append(answer == qa[f'a{qa["answer_idx"]}'])
#             answer = answer_question(model_2, qa)
#             acc2.append(answer == qa[f'a{qa["answer_idx"]}'])
#         except ValueError:
#             print(qa)


print(len(acc1), round(sum(acc1)/len(acc1), 4))
print(len(acc2), round(sum(acc2)/len(acc2), 4))
