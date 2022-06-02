import argparse
from pathlib import Path
import random
from re import L
from tqdm import tqdm
import os
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from accelerate import Accelerator
from datasets import load_metric
from transformers import (
    BertForSequenceClassification,
    BertTokenizerFast,
    DataCollatorWithPadding,
    AutoTokenizer,
    set_seed
)

from nni.algorithms.compression.pytorch.pruning import HardwareAwarePruner
from nni.compression.pytorch.pruning import BalancedPruner
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

model_name = 'bert-base-cased'
max_seq_length = 128
train_batch_size = 32
eval_batch_size = 32
root_path = './data'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

finetune_epoch = {
    "cola": 20,
    "mnli": 5,
    "mrpc": 20,
    "qnli": 10,
    "qqp": 5,
    "rte": 20,
    "sst2": 20,
    "stsb": 20,
    "wnli": 20
}

def trainer(model, optimizer, train_dataloader):
    model.train()
    for batch in tqdm(train_dataloader):
        batch.to(device)
        optimizer.zero_grad()
        outputs = model(**batch)
        outputs.loss.backward()
        optimizer.step()

def evaluator(model, metric, is_regression, eval_dataloader):
    model.eval()
    for batch in tqdm(eval_dataloader):
        batch.to(device)
        outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
        metric.add_batch(
            predictions=accelerator.gather(predictions),
            references=accelerator.gather(batch["labels"]),
        )
    return metric.compute()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the pretrained bert on GLUE task.')
    parser.add_argument('task_name', type=str)
    parser.add_argument('--seed', type=int, default=1024)
    args = parser.parse_args()

    task_name = args.task_name
    num_labels = 3 if task_name == 'mnli' else 2
    seed = args.seed

    is_regression = True if task_name == 'stsb' else False

    setup_seed(seed)

    train_dataset = torch.load(Path(root_path, 'splited-glue-data/resplit', '{}_train_dataset.pt'.format(task_name)))
    validate_dataset = torch.load(Path(root_path, 'splited-glue-data/resplit', '{}_validate_dataset.pt'.format(task_name)))
    test_dataset = torch.load(Path(root_path, 'splited-glue-data/resplit', '{}_test_dataset.pt'.format(task_name)))

    accelerator = Accelerator()
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))

    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=train_batch_size)
    validate_dataloader = DataLoader(validate_dataset, collate_fn=data_collator, batch_size=eval_batch_size)
    test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=eval_batch_size)

    # model = torch.load(Path(root_path, 'pretrained_glue_models_bert-base-cased', 'pretrained_model_{}.pt'.format(task_name))).to(device)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).cuda()
    optimizer = Adam(model.parameters(), lr=2e-5)

    metric = load_metric("glue", task_name)

    val_metric = evaluator(model, metric, is_regression, validate_dataloader)
    test_metric = evaluator(model, metric, is_regression, test_dataloader)


    # import ipdb; ipdb.set_trace()
    data_dir = f"bert_{task_name}_pretrained_cks"
    os.makedirs(data_dir, exist_ok=True)

    # _, model, masks, _, _ = pruner.get_best_result()
    for epoch in range(finetune_epoch[task_name]):
        trainer(model, optimizer, train_dataloader)
        val_metric = evaluator(model, metric, is_regression, validate_dataloader)
        test_metric = evaluator(model, metric, is_regression, test_dataloader)
        with open(os.path.join(data_dir, 'result.csv'), 'a+') as f:
            f.write('{}, {}, {}, {}, {}, {}\n'.format(task_name, seed, sparsity, epoch, val_metric, test_metric))
    weight_path = os.path.join(data_dir, 'weight.pth')
    torch.save(model.state_dict(), weight_path)


