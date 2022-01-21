#!/bin/bash

python train.py \
    --dir_dir=data/smiles/RCC_key1 \
    --max_epochs=10000 \
    --model_dir=model/smiles/380k_pretrain \
    --dict=word2idx.json
