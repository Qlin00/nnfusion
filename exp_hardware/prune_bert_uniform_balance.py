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
from torch.optim.lr_scheduler import CosineAnnealingLR
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
    "mnli": 50,
    "mrpc": 20,
    "qnli": 10,
    "qqp": 10,
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
    parser.add_argument('--sparsity', type=float, default=0.5)
    parser.add_argument('--alignn', type=int, default=8)
    parser.add_argument('--cks', type=str, default=None)
    parser.add_argument('--outdir', type=str, default=None)
    args = parser.parse_args()
    if 'AMLT_OUTPUT_DIR' in os.environ:
        args.outdir = os.environ['AMLT_OUTPUT_DIR']

    task_name = args.task_name
    num_labels = 3 if task_name == 'mnli' else 2
    seed = args.seed
    sparsity = args.sparsity
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
    if args.cks:
        model.load_state_dict(torch.load(args.cks))
    optimizer = Adam(model.parameters(), lr=2e-5)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=finetune_epoch[task_name])
    metric = load_metric("glue", task_name)
    # import ipdb; ipdb.set_trace()
    val_metric = evaluator(model, metric, is_regression, validate_dataloader)
    test_metric = evaluator(model, metric, is_regression, test_dataloader)
    print('Accuracy of initial checkpoints', test_metric)
    def hardware_evaluator(model):
        result = evaluator(model, metric, is_regression, test_dataloader)
        return result['accuracy']
    # with Path('./result.csv').open('a+') as f:
    #     f.write('{}, {}, {}, {}, {}, {}\n'.format(task_name, seed, sparsity, -1, val_metric, test_metric))

    op_names = []
    op_names.extend(["bert.encoder.layer.{}.attention.self.query".format(i) for i in range(0, 12)])
    op_names.extend(["bert.encoder.layer.{}.attention.self.key".format(i) for i in range(0, 12)])
    op_names.extend(["bert.encoder.layer.{}.attention.self.value".format(i) for i in range(0, 12)])
    op_names.extend(["bert.encoder.layer.{}.attention.output.dense".format(i) for i in range(0, 12)])
    op_names.extend(["bert.encoder.layer.{}.intermediate.dense".format(i) for i in range(0, 12)])

    # config_list = [{'op_types': ['Linear'], 'op_names': op_names, 'sparsity': sparsity}]
    config_list = [{'op_types': ['Linear'], 'sparsity': sparsity, 'op_names':op_names}, {'exclude':True, 'op_names':['classifier']}]
    # import ipdb; ipdb.set_trace()
    data_dir = f"bert_{task_name}_{sparsity}_uniform_align_n_{args.alignn}"
    if args.outdir is not None:
        data_dir = args.outdir
    os.makedirs(data_dir, exist_ok=True)
    # pruner = HardwareAwarePruner(model, config_list, hardware_evaluator, align_n_set=[1,2,4,8,16,32], experiment_data_dir=data_dir, need_sort=False)
    pruner = BalancedPruner(model, config_list, align_n=[args.alignn, 1], balance_gran=[1, 32])
    model, masks = pruner.compress()
    # import ipdb; ipdb.set_trace()
    # _, model, masks, _, _ = pruner.get_best_result()
    for epoch in range(finetune_epoch[task_name]):
        trainer(model, optimizer, train_dataloader)
        val_metric = evaluator(model, metric, is_regression, validate_dataloader)
        test_metric = evaluator(model, metric, is_regression, test_dataloader)
        with open(os.path.join(data_dir, 'result.csv'), 'a+') as f:
            f.write('{}, {}, {}, {}, {}, {}\n'.format(task_name, seed, sparsity, epoch, val_metric, test_metric))
        lr_scheduler.step()
        print('Learning rate: ', lr_scheduler.get_last_lr())
    weight_path = os.path.join(data_dir, 'weight.pth')
    mask_path = os.path.join(data_dir, 'mask.pth')
    # import pdb; pdb.set_trace()
    torch.save(model.state_dict(), weight_path)
    torch.save(masks, mask_path)
    # pruner.export_model(weight_path, mask_path)
