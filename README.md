# nlp-hw2

This repo contains code for A2 of Natural Language Processing. The assignment is to fit an n-gram model to the One Billion Language Modeling Benchmark dataset.

## System Requirements

The code runs on Python 3, with only `numpy` as an external dependence.

## Data

The preprocessing code in `preprocess.py` assumes a folder `data/` containing the files `1b_benchmark.train.tokens`, `1b_benchmark.dev.tokens`, and `1b_benchmark.test.tokens`, which are not included in the repo. However, processesed data files are saved in `train_sequence.pkl`, `dev_sequence.pkl`, and `test_sequence.pkl`.

## Instructions

The `lm.py` file contains the implementation of the n-gram model. `train.py` fits and saves the model, while `test.py` loads the saved model and evaluates perplexity. `generate.py` can be used to load the saved model and generate sentences!
