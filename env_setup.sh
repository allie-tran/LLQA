#!/bin/bash
# run source env_setup.sh
export DISTRACTOR_PATH=distractor
export DISTRACTOR_DATA_PATH=models/distractor.pt

export TFIDF_PATH=models/knowledge/

export QUESTION_GEN_PATH=models/whword_gen/
export MULTIPLE_ANSWERER_PATH=models/s2s_answerer/
export MULTIPLE_ANSWERER_PATH_KNOWLEDGE=models/s2s_retrained/

export MNT=/home/tlduyen/LQA/mnt/
