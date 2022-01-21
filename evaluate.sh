#!/bin/bash

python evaluate.py \
    --data_dir=data/smiles/RCC_key1 \
    --out_dir=out/smiles/RCC_key1 \
    --batch_size=32 \
    --model_dir=model/smiles/RCC_key1 \
    --dict_=word2idx.json
