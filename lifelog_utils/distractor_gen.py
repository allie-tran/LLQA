from transformers import AutoTokenizer, BartForConditionalGeneration

class DistractorGenerator():
    def __init__(self):
        self.loaded = False


    def load(self):
        if self.loaded:
            return
        model_names = ["voidful/bart-distractor-generation-both"]
        self.tokenizers = [AutoTokenizer.from_pretrained(name) for name in model_names]
        self.distractors = [BartForConditionalGeneration.from_pretrained(name) for name in model_names]
        self.loaded = True
        print("Finished loading distractor generator.")

    def to_text(self, model_index, outputs, num=3):
        distractors = []
        for i in range(num):
            distractors.append(self.tokenizers[model_index].decode(
                outputs[i], skip_special_tokens=True))
        return distractors


    def generate(self, context, question, answer):
        doc = context + '</s>' + question + '</s>' + answer
        all_options = []
        for i in range(len(self.tokenizers)):
            input_ids = self.tokenizers[i](doc, return_tensors="pt").input_ids
            outputs = self.distractors[i].generate(
                input_ids=input_ids, num_beams=12, num_beam_groups=3, num_return_sequences=3, diversity_penalty=0.5, length_penalty=0.2)
            all_options.extend(self.to_text(i, outputs, num=3))
        return all_options
