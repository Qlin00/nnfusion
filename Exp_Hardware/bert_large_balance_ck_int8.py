import argparse
import glob
import json
import logging

import os
import random
import math

import numpy as np
import torch
# from torch.functional import norm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from emmental import MaskedBertConfig, MaskedBertForSequenceClassification
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertModel,
    BertForSequenceClassification,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)
from transformers import glue_compute_metrics as compute_metrics
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors
from emmental.modules.masked_nn import MaskedLinear
import nni
import torch
import sys
import os
from nni.compression.pytorch.pruning import LevelPruner
from sparta.common.utils import export_tesa, export_tesa_debug, generate_balance_cfg_int8, generate_balance_pattern
device = torch.device('cpu')
remain_n = 8
total_m = 32
align = 64
sparsity_ratio = 1 - remain_n / total_m
outdir = f'balance_bert_large_n_{remain_n}_m_{total_m}_align{align}'
dummy_input = torch.load('dummy_input.pth', map_location=device)
data = (dummy_input['input_ids'].to(device), dummy_input['attention_mask'].to(device), dummy_input['token_type_ids'].to(device))
norm_model = BertModel.from_pretrained("bert-large-uncased")

cfg_list = []
for name, module in norm_model.named_modules():
    if name != 'classifier':
        cfg_list.append({'sparsity':sparsity_ratio, 'op_types':['Linear'], 'op_names':[name]})

# pruner = LevelPruner(norm_model, cfg_list, balance_gran=[1, 32], block_sparse_size=[8, 1], mode='balance')
pruner = LevelPruner(norm_model, cfg_list)
# get the propagated mask
pruner.compress()
pruner.export_model('./weight.pth', './mask.pth')
# generate random mask
pruner._unwrap_model()
mask =  torch.load('./mask.pth')

for name in mask:
    n, k = mask[name]['weight'].size()
    mask[name]['weight'] = generate_balance_pattern(n, k, total_m, align, sparsity_ratio).to(mask[name]['weight'].device)
for name, module in norm_model.named_modules():
    if isinstance(module, torch.nn.Linear):
        module.weight.data[:] = 1
export_tesa(norm_model.cpu(), data, outdir, mask)

generate_balance_cfg_int8(outdir, align, total_m, sparsity_ratio)
