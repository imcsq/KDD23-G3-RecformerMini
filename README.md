# Implementation of Recformer

## Operating System: Linux

## Environment Setup
1. Run `pip3 install -r requirements.txt` to install necessary Python packages.

## Data Preparing
1. Run `cd pretrain_data; python3 preprocess.py -d -m -s; cd ..` for pretrain data pre-processing, where `-d`, `-m`, `-s` are for data downloading, meta data extraction, and interaction sequence generation respectively.
2. Run `cd finetune_data; python3 preprocess.py -d -m -s; cd ..` for finetune data pre-processing, where `-d`, `-m`, `-s` are for data downloading, meta data extraction, and interaction sequence generation respectively.

## Conduct Experiment
1. Run `bash 1-pretrain.sh` to perform pre-training.
2. Run `python 2-convert_pretrained_ckpt.py` to transform the lightning framework model to torch model.
3. Run `bash 3-finetune.sh` to perform fine-tuning on 6 datasets specified in the original work.

## Description of Source Files
1. `pretrain_data/preprocess.py` is to pre-process the pre-training dataset.
2. `finetune_data/preprocess.py` is to pre-process the fine-tuning datasets.
3. `lightning_dataloader.py` is the dataloader for pre-training.
4. `dataloader.py` is the dataloader for fine-tuning.
5. `collator.py` is to collects and processes data into batches and the output can be directly feed to pretraining finetuning evaluation and testing the model.
6. `recformer/tokenization.py` tokenizes the item sequences by token ids, token position ids, token type ids item position ids.
6. `recformer/models.py` is the implementation of the base model with 4 embedding layers extended from Longformer, and the models for pretraining and prediction.
7. `lightning_litwrapper.py` is the lightning module wrapper for performing easier pretraining with torch_lightning.
8. `lightning_pretrain.py` is the script to perform pretraining with torch_lightning framework.
9. `finetune.py` is to fine-tune the pretrained model.
