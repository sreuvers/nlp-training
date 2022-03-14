import os
os.environ['LD_LIBRARY_PATH']='/usr/local/lib'
from transformers import Trainer, TrainingArguments, BertForSequenceClassification, DistilBertForSequenceClassification, set_seed, HfArgumentParser
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

set_seed(2021)

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


def train_BERT(model, args):
    """
    This contains everything that must be done to train our models
    """
    train_path = args.path_data + "/train.pt"
    val_path = args.path_data + "/validation.pt"

    train_dataset = torch.load(train_path)
    val_dataset = torch.load(val_path)

    training_args = TrainingArguments(
        output_dir=args.path_output + 'results/',
        num_train_epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        evaluation_strategy="epoch",
        eval_accumulation_steps=10,
        save_strategy="epoch",
        weight_decay=0.01,  # strength of weight decay
        logging_dir=args.path_output + 'logs/',  # directory for storing logs
        load_best_model_at_end=True,
        learning_rate=args.learning_rate,
        metric_for_best_model="f1",
        overwrite_output_dir =True,
        per_device_train_batch_size=args.train_batch_size,  # batch size per device during training
        per_device_eval_batch_size=args.eval_batch_size,  # batch size for evaluation
        # report_to="wandb",  # enable logging to W&B
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

def _mp_fn(index,args):
    device = xm.xla_device()
    # We wrap this
    model = WRAPPED_MODEL.to(device, )
    train_BERT(model, args)

@dataclass
class TrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    train_file: Optional[str] = field(default=None, metadata={"help": "The input tokenized training data file"})
    validation_file: Optional[str] = field(default=None, metadata={"help": "The input tokenized validation data file"})
    name_run: Optional[str] = field(default=None, metadata={"help": "The name of the run"})
    path_output: Optional[str] = field(default=None, metadata={"help": "The output path"})
    path_data: Optional[str] = field(default=None, metadata={"help": "The data path"})


    # TODO: properly import model and tokenizer
    path_model: Optional[str] = field(default=None, metadata={"help": "The model path"})
    path_tokenizer: Optional[str] = field(default=None, metadata={"help": "The tokenizer path"})

    epochs: Optional[int] = field(default=3, metadata={"help": "The input tokenized training data file"})
    train_batch_size: Optional[int] = field(default=32, metadata={"help": "The input tokenized validation data file"})
    eval_batch_size: Optional[int] = field(default=32, metadata={"help": "The name of the run"})
    learning_rate: Optional[float] = field(default=5e-5, metadata={"help": "The output path"})
    warmup_steps: Optional[int] = field(default=500, metadata={"help": "The output path"})

if __name__ == "__main__":
    parser = HfArgumentParser((TrainingArguments))
    args = parser.parse_args_into_dataclasses()[0]
    args.path_output = args.path_output + args.name_run

    os.environ["WANDB_DISABLED"] = "true"

    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

    model.train()

    WRAPPED_MODEL = xmp.MpModelWrapper(model)

    xmp.spawn(_mp_fn, args = (args,), nprocs=8, start_method="fork")

