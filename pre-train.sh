#!/bin/bash

python train.py \
    --dir_dir=data/smiles/380k_pretrain \
    --max_epochs=10000 \
    --dict=word2idx.json
