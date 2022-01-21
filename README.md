# New application of natural language process (NLP) for chemist: Predicting intermediate and providing effective direction for mechanism inference

This is the code for "New application of natural language process (NLP) for chemist: Predicting intermediate and providing effective direction for mechanism inference" paper.

## Setup

```bash
conda env create -n ABCD -f environment.yml
conda activate ABCD
```

## Dataset

USPTO_380k was used as pretraining dataset: `data/smiles/380k_pretrain`
Raw radical cascade cyclization dataset: `data/smiles/RCC.txt`

```bash
unzip data.zip
```

## Quickstart

```bash
mkdir -p out/smiles/380k_pretrain \
    out/smiles/RCC_key1 \
    model/smiles/380k_pretrain \
    model/smiles/RCC_key1
```

### Pretrain on 380k

```bash
bash pre-train.sh
```

Move `checkpoints` and all wheights generated in this running to the model dir `model/smiles/380k_pretrain`

### Finetuning on radical cascade cyclization dataset

We only demonstrated the prediction of key intermediates in this demo

```bash
bash train.sh
```

Move `checkpoints` and all wheights generated in this running to the model dir `model/smiles/RCC_key_1`

### Evaluating

```bash
bash evaluate.sh
```

### Get accuracy

copy the target (labels) file to `out` dir as well as the results of evaluating above

```bash
cp data/smiles/RCC_key1/test.target out/smiles/RCC_key1
mv greedy.out out/smiles/RCC_key1
```

Then `accuracy.txt` was obtained

```bash
python get_acc.py
```
