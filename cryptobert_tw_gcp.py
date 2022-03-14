import os
os.environ['LD_LIBRARY_PATH']='/usr/local/lib'
from transformers import Trainer, TrainingArguments, BertForSequenceClassification, DistilBertForSequenceClassification, set_seed
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
set_seed(2021)

TRAIN = True
FIRST_RUN = True
FIRST_PREDICT = True
COLAB = False

if COLAB:
  path_data = '/content/drive/MyDrive/Thesis/CryptoBERT/datasets/old/test_train_10000_val_1000'
  path_output = '/content/drive/MyDrive/Thesis/CryptoBERT/fineTuned_small'
else:
  path_data = '/home/bijlesjvl/data/test_train_10000_val_1000'
  path_output = '/home/bijlesjvl/model/fineTuned_small'

class TweetsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def compute_metrics(pred):
    """
    Compute metrics for Trainer
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # _, _, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro")

    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        # 'macro f1': macro_f1,
        'precision': precision,
        'recall': recall
    }


def train_BERT(model, epochs=5, train_batch_size=32, eval_batch_size=32, learning_rate=5e-5, warmup_steps=1000,
               folder='bal-bal_train_500000_val_25000'):
    """
    This contains everything that must be done to train our models
    """
    train_path = path_data + "/train.pt"
    val_path = path_data + "/validation.pt"

    train_dataset = torch.load(train_path)
    val_dataset = torch.load(val_path)

    training_args = TrainingArguments(
        output_dir=path_output + 'results/',
        num_train_epochs=epochs,
        warmup_steps=warmup_steps,
        evaluation_strategy="epoch",
        eval_accumulation_steps=10,
        save_strategy="epoch",
        weight_decay=0.01,  # strength of weight decay
        logging_dir=path_output + 'logs/',  # directory for storing logs
        load_best_model_at_end=True,
        learning_rate=learning_rate,
        metric_for_best_model="f1",
        per_device_train_batch_size=train_batch_size,  # batch size per device during training
        per_device_eval_batch_size=eval_batch_size,  # batch size for evaluation
        report_to="wandb",  # enable logging to W&B
        run_name="Finetuning on TPU"  # name of the W&B run (optional)
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    trainer.place_model_on_device = False
    trainer.train()

# !pip install -U numpy
def _mp_fn(index):
    device = xm.xla_device()
    # We wrap this
    model = WRAPPED_MODEL.to(device)
    train_BERT(model)

xmp.spawn(_mp_fn, nprocs=8,start_method="fork")

wandb.finish()