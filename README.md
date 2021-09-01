# New application of natural language process (NLP) for chemist: Predicting intermediate and providing effective direction for mechanism inference

This is the code for " New application of natural language process (NLP) for chemist: Predicting intermediate and providing effective direction for mechanism inference" paper. 

## Environment

```python
conda env create -n pred_intermdt -f environment.yml
codna activate red_intermdt
```

## Dataset

The dataset for pretraining were provided in ```data/pre-train dataset``` file. 

The train, validation and test of radical cascade cyclization intermediate prediction data set were provided in ```data/radical cascade cyclization dataset``` file.

```python
# if data is zipped, unpack it
unzip data.zip
```
## Quickstart

### Step 1. train the model on radical cascade cyclization data set

```python
# get the transformer-baseline model
# modify the input dir path manually in train_transformer.py
python train_transformer_train_radical.py
```

### Step 2: train the model on general chemical reaction data set

```python
#　get a pretrained model
# modify the input dir path manually in train_transformer.py
python train_transformer_380k.py  
```

### Step 3: train the pretrained model on radical cascade cyclization data set

```python
# after pretraining, get back to original data path
# manually midification is required in train_transformer.py
python train_transformer_train_radical.py
```

### Step 4: test

```python
python test.py
```



