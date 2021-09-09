#!/usr/bin/env bashmnt/

export CUDA_VISIBLE_DEVICES=0,1

python -u translate.py \
    -model=/mnt/DATA/tlduyen/LQA/data/model/jul2_model_step_45000.pt \
    -data=data/race_test_updated.json \
    -output=data/pred/jul2_model_step_45000.txt \
    -share_vocab \
    -block_ngram_repeat=1 \
    -replace_unk \
    -batch_size=1 \
    -beam_size=50 \
    -n_best=50 \
    -gpu=0
