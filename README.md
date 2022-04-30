# SemAttack

This is the official code base for the NAACL 2022 paper, "[SemAttack: Natural Textual Attacks via Different Semantic Spaces](https://openreview.net/forum?id=03-jwvIDYf)".

This repo contains code to attack both English and Chinese models. We put the code in seperate folders.

You may use our code to attack other tasks or other datasets.

## BERT Embedding Space

We follow the process in paper ["Visualizing and Measuring the Geometry of BERT"](https://arxiv.org/abs/1906.02715) ([GitHub](https://github.com/PAIR-code/interpretability)) to calculate word embeddings. The embedding space is stored as an N * 768 tensor, where N is the number of total embeddings.

## Model Training

Use `train.py` to train your models. We also provide our pre-trained models [here](https://drive.google.com/drive/folders/1pjPlxGWVbPpWNMueOF54f7oQP_hoJcRY?usp=sharing).

## Attack

Use `attack.py` to attack pre-trained models. You may check the code for more details. You may also try different semantic perturbation functions and different attack parameters.