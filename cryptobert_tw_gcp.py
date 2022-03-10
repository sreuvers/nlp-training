# -*- coding: utf-8 -*-
"""cryptoBERT_TW GCP.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1wzBUg9vKFaaolz5h43M0WzBW31RlA_4Z

Load files from Google Drive and create train, test and validation part
"""



# Commented out IPython magic to ensure Python compatibility.
# %%capture
# 
# curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py
# # !python pytorch-xla-env-setup.py --version $VERSION
# python3 pytorch-xla-env-setup.py --version 1.9 --apt-packages libomp5 libopenblas-dev
# # !python3 pytorch-xla-env-setup.py --version nightly --apt-packages libomp5 libopenblas-dev
# 
# pip install wandb
# pip install --upgrade wandb[service]
# pip install transformers
# 
#
import os 
os.environ['LD_LIBRARY_PATH']='/usr/local/lib'

import torch_xla.distributed.xla_multiprocessing as xmp

from transformers import Trainer, TrainingArguments, BertForSequenceClassification, DistilBertForSequenceClassification, set_seed
set_seed(2021)
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


import pprint
import os 
import torch


os.environ['LD_LIBRARY_PATH']='/usr/local/lib'


TRAIN = True
FIRST_RUN = True
FIRST_PREDICT = True

path_data = '/home/bijlesjvl/data/test_train_10000_val_1000'
path_output = '/home/bijlesjvl/model/fineTuned_small'

# Commented out IPython magic to ensure Python compatibility.

import wandb

wandb.login()
# %env WANDB_PROJECT=cryptoBERT
# %env WANDB_LOG_MODEL=true

wandb.require(experiment="service")

wandb.init(project="cryptoBERT", entity="srnl",resume="allow")

name_sweep = options[0]['folder']

sweep_config = {
    'name' : name_sweep,
    'method': 'grid', #grid, random, bayesian
    'metric': {
      'name': 'eval/accuracy',
      'goal': 'maximize'   
    },
    'parameters': {

        'learning_rate': {
            'values': [5e-5, 3e-5, 1e-5, 3e-4]
        },
        'train_batch_size': {
            'values': [32, 64,128]
        },
            'eval_batch_size': {
            'values': [32, 64,128]
        },
        'epochs': {
            'values': [2,3,4]
        },
        'tokenizer_max_len': {'value': 128},
    }
}

pprint.pprint(sweep_config)

sweep_id = wandb.sweep(sweep_config, project="cryptoBERT")



# from google.colab import files
# !cp -r /content/drive/MyDrive/Thesis/CryptoBERT/interface_queue.py /usr/local/lib/python3.7/dist-packages/wandb/sdk/interface/interface_queue.py

model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
# model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

model.train()

WRAPPED_MODEL = xmp.MpModelWrapper(model)


def compute_metrics(pred):
    """
    Compute metrics for Trainer
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    #_, _, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro")
    
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        #'macro f1': macro_f1,
        'precision': precision,
        'recall': recall
    }



def train_BERT(model, epochs=5, train_batch_size=32,eval_batch_size=32, learning_rate = 5e-5, warmup_steps=1000, folder = 'bal-bal_train_500000_val_25000'):
    """
    This contains everything that must be done to train our models
    """

    print("Loading datasets... ", end="")
    
    train_path = path_data + "/train.pt"
    val_path = path_data + "/validation.pt"

    train_dataset = torch.load(train_path)
    val_dataset = torch.load(val_path)

 

    training_args = TrainingArguments(
        output_dir= path_output + 'results/',
        num_train_epochs=epochs,
        warmup_steps=warmup_steps,
        evaluation_strategy="epoch",
        eval_accumulation_steps = 10,
        save_strategy="epoch",
        weight_decay=0.01,                    #strength of weight decay
        logging_dir=path_output + 'logs/',            # directory for storing logs
        load_best_model_at_end=True,
        learning_rate = learning_rate,
        metric_for_best_model="f1",
        per_device_train_batch_size=train_batch_size,  # batch size per device during training
        per_device_eval_batch_size=eval_batch_size,   # batch size for evaluation
        report_to="wandb",  # enable logging to W&B
        run_name= "Finetuning on TPU"  # name of the W&B run (optional)
    )


    results = []

    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    trainer.place_model_on_device = False
    trainer.train()

import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

def _mp_fn(index,config):
    device = xm.xla_device()
    # We wrap this 
    model = WRAPPED_MODEL.to(device)
    train_BERT(model, epochs=config.epochs, train_batch_size=config.train_batch_size, eval_batch_size=config.eval_batch_size,learning_rate = config.learning_rate)

def train(config=None):
    # Initialize a new wandb run
    with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config
        xmp.spawn(_mp_fn, args = (config,),nprocs=8, start_method="fork")

wandb.agent(sweep_id, function=train, count = 10)

# wandb.agent(sweep_id, train, count=10)

wandb.finish()

"""Predict new data"""



# path_output = '/content/drive/MyDrive/Thesis/CryptoBERT/'
# path_data_raw = path_output + 'variables/text_twits.pkl'
# path_data = path_output + 'datasets/'
 
# train_path = path_data + "train.pt"
# val_path = path_data + "validation.pt"

# train_dataset = torch.load(train_path)

# sentence_lengths = []
# for text in train_texts:
#   sentence_lengths.append(len(text.split(" ")))

# max(sentence_lengths)
# min(sentence_lengths)

# import matplotlib.pyplot as plt 

# # default bins = 10
# plt.hist(sentence_lengths,bins = 30)

# plt.show()

# larger_65 = sum(i > 60 for i in sentence_lengths)/len(sentence_lengths)

# val_dataset = torch.load(val_path)
# print(len(val_dataset))
# for i in range(0,10):
#   review = val_dataset[i]
#   ids = review['input_ids']
#   print(list(ids))



# 50000# ----- 3. Predict -----#
# import numpy as np
# import pickle
# import pandas as pd
# from torch.utils.data import DataLoader
# from sklearn.metrics import classification_report
# import textwrap as tr
# import torch.nn.functional as F 

# pred_path = path_output + "results/predicted/prediction.pickle"
# model_path = path_output + "results/trained_model"
# test_path = path_data + "test.pt"
# test_dataset = torch.load(test_path)
# model = DistilBertForSequenceClassification.from_pretrained(model_path, num_labels=2)
# model.eval()
# with open(path_data_raw, 'rb') as f:
#   train_texts, train_labels, test_texts, test_labels = pickle.load(f)

# # Define test trainer
# test_trainer = Trainer(model)


# if FIRST_PREDICT:
#   # Make prediction
#   predictions, label_ids, metrics = test_trainer.predict(test_dataset,return_all_scores=True)
#   with open(pred_path, 'wb') as handle:
#       pickle.dump([predictions, label_ids, metrics], handle, protocol=pickle.HIGHEST_PROTOCOL)
# else:
#   with open(pred_path, 'rb') as f:
#       predictions, label_ids, metrics = pickle.load(f)

# # Preprocess raw predictions
# predictions = np.argmax(predictions, axis=1)
# predictions = ["Positive" if item == 1 else "Negative" for item in predictions]
# labels = ["Positive" if item == 1 else "Negative" for item in label_ids]

# # Make contigency table 

# # Create the evaluation report.
# evaluation_report = classification_report(labels, predictions)
# # Show the evaluation report.
# print(evaluation_report)


# # Show some examples
# # shuffled_dataset = torch.utils.data.Subset(test_dataset, torch.randperm(len(test_dataset)).tolist())
# # dataloader = DataLoader(shuffled_dataset, batch_size=1, num_workers=1, shuffle=False)
# # print(dataloader)





# for i in range(0,len(test_dataset)):
#   review = test_dataset[i]
#   ids = review['input_ids']
#   tokens = tokenizer.convert_ids_to_tokens(ids,skip_special_tokens = True)
#   string = tokenizer.convert_tokens_to_string(tokens)
#   print(tr.fill(test_texts[i], width=70))
#   print("\nTrue label: \t %s \nPredicted label: %s \n"  % (labels[i], predictions[i]))
#   if i == 5:
#     break

# model = DistilBertForSequenceClassification.from_pretrained(model_path, num_labels=2,output_hidden_states=True)

# for i in range(1,10):
#   review = test_dataset[i]
#   token_ids = review['input_ids'].unsqueeze(0)
#   attention_mask = review['attention_mask'].unsqueeze(0)
#   output = model(token_ids, attention_mask = attention_mask)
#   hiddenStates = output.hidden_states

#   tensors = output[0][0]                                             
#   print(list(F.softmax(tensors, dim = 0).detach().numpy())) #use softmax as only one label is possible

# hiddenStates
# layer_i = 6 #last layer before the classifier (12!)
# batch_i = 0 # only one input in the batch
# token_i = 0 # first token corresponding to CLS 

# vec = hiddenStates[layer_i][batch_i][token_i]
# print(vec)