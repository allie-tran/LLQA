## Lifelog data
Download the lifelog dataset here: http://lsc.dcu.ie/.

## Requirements
Create new environment
```
conda env create -f environment.yml
```
Install Apex:
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

## Generating QA pairs
Example code to generate QA pairs is in `generate_questions.py`.
```bash
$ python generate_questions.py
```

The QA pairs are in the folder `published`. Example data:
```javascript
{
    "question": "What is the lifelogger doing?",
    "answers": ["Watching TV .", "Playing football .", "Doing homework .", "Drinking water ."],
    "answer_idx": 0,
    "date": "2015-03-14",
    "start_time": "2015/03/14 09:32:00+00",
    "end_time": "2015/03/14 09:36:00+00"
}
```
We also include the annotated description in `published/descriptions.jsonl`.

## Credits:
- Original `s2s-ft` codes: https://github.com/microsoft/unilm/tree/master/s2s-ft
