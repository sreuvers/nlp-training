import os
os.environ['LD_LIBRARY_PATH']='/usr/local/lib'
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification, set_seed, HfArgumentParser
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple
from datasets import load_from_disk, load_dataset
from torch.nn import CrossEntropyLoss

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
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro")
    acc = accuracy_score(labels, preds)
    loss_fct = CrossEntropyLoss(weight=torch.tensor([float(args.weights_1), float(args.weights_2)]))
    loss = loss_fct(torch.from_numpy(pred.predictions).view(-1, 2), torch.from_numpy(labels).view(-1))

    return {
        'accuracy': acc,
        'f1': f1,
        # 'macro f1': macro_f1,
        'precision': precision,
        'recall': recall,
        'true loss': loss
    }

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 3 labels with different weights)
        loss_fct = CrossEntropyLoss(weight=torch.tensor(model.custom_weights))
        loss = loss_fct(logits.view(-1, 2), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


def train_BERT(model, args):
    """
    This contains everything that must be done to train our models
    """
    data = load_from_disk(args.path_data + 'tokenized')

    train_dataset = data['train']
    val_dataset = data['test']

    training_args = TrainingArguments(
        num_train_epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        eval_accumulation_steps=10,
        save_strategy="epoch",
        weight_decay=0.1,  # strength of weight decay
        logging_dir=args.path_output + '/logs/',  # directory for storing logs
        learning_rate=args.learning_rate,
        metric_for_best_model="f1",
        per_device_train_batch_size=args.train_batch_size,  # batch size per device during training
        per_device_eval_batch_size=args.eval_batch_size,  # batch size for evaluation
        output_dir=args.path_output + '/results/',
        overwrite_output_dir=True,
        report_to="tensorboard",
        run_name="Finetuning on TPU",  # name of the W&B run (optional)
        evaluation_strategy="steps",
        eval_steps = 100,
        logging_strategy="steps",
        logging_steps=100
    )

    trainer = CustomTrainer(
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
class TrainingArgumentsInput:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    path_output: Optional[str] = field(default=None, metadata={"help": "The output path"})
    path_data: Optional[str] = field(default=None, metadata={"help": "The data path"})

    model_name: Optional[str] = field(default=None, metadata={"help": "The model name"})

    # TODO: properly import model and tokenizer
    path_model: Optional[str] = field(default=None, metadata={"help": "The model path"})
    weights_1: Optional[float] = field(default=1, metadata={"help": "The weight of class 0"})
    weights_2: Optional[float] = field(default=1, metadata={"help": "The weight of class 1"})
    epochs: Optional[int] = field(default=3, metadata={"help": "The input tokenized training data file"})
    train_batch_size: Optional[int] = field(default=32, metadata={"help": "The input tokenized validation data file"})
    eval_batch_size: Optional[int] = field(default=32, metadata={"help": "The name of the run"})
    learning_rate: Optional[float] = field(default=5e-5, metadata={"help": "The output path"})
    warmup_steps: Optional[int] = field(default=500, metadata={"help": "The output path"})




def tokenize_function(examples):
    # Remove empty lines
    examples = [line for line in examples if len(line) > 0 and not line.isspace()]
    return tokenizer_custom(
        examples,
        return_special_tokens_mask=True,
        padding="max_length",
        truncation=True,
        max_length=128,
    )

def getDataset(args):
    datasets = load_from_disk(args.path_data)

    column_names = datasets["train"].column_names
    text_column_name = "tweet" if "tweet" in column_names else column_names[0]
    keep_names = ['labels']
    column_names_remove = [item for item in column_names if item not in keep_names]
    tokenized_datasets = datasets.map(
        lambda examples: tokenize_function(examples[text_column_name]),
        batched=True,
        remove_columns=column_names_remove)
    tokenized_datasets.save_to_disk(args.path_data + 'tokenized')


if __name__ == "__main__":
    # wandb.require(experiment="service")
    parser = HfArgumentParser((TrainingArgumentsInput))
    args = parser.parse_args_into_dataclasses()[0]
    os.environ["WANDB_DISABLED"] = "true"

    tokenizer_custom = AutoTokenizer.from_pretrained(args.path_model)
    print("TOKENIZING DATASET...")
    getDataset(args)
    print("TOKENIZED DATASET!")

    if "Crypto" in args.model_name:
        print("LOAD CUSTOM MODEL FROM FLASK")
        model = AutoModelForSequenceClassification.from_pretrained(args.path_model,num_labels = 2, from_flax=True)
    else:
        print("LOAD MODEL FROM HUGGINGFACE")
        model = AutoModelForSequenceClassification.from_pretrained(args.path_model)
    model.custom_weights =[float(args.weights_1), float(args.weights_2)]
    model.train()

    WRAPPED_MODEL = xmp.MpModelWrapper(model)

    print("START TRAINING...")
    xmp.spawn(_mp_fn, args = (args,), nprocs=8, start_method="fork")
    print("FINISHED TRAINING!")
    raise SystemExit()


