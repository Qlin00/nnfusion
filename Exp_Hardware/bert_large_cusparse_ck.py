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
from sparta.common.utils import export_tesa, export_tesa_debug, generate_cusparse_sparse_cfg
device = torch.device('cpu')
remain_n = 16
total_m = 32
align = 1
sparsity_ratio = 1 - remain_n / total_m
in_dir = f'balance_bert_large_n_{remain_n}_m_{total_m}_align{align}'
out_dir = f'cusparse_bert_large_n_{remain_n}_m_{total_m}_align{align}'
assert os.path.exists(in_dir), f'{in_dir} does not exists'

norm_model = BertModel.from_pretrained("bert-large-uncased")
state_path = os.path.join(in_dir, 'state_dict.pth')
tesa_path = os.path.join(in_dir, 'tesa')
id_map_path = os.path.join(in_dir, 'tesaid_2_names')
model_path = os.path.join(in_dir, 'model_tesa.onnx')

generate_cusparse_sparse_cfg(tesa_path, state_path, id_map_path, out_dir)
os.system(f"cp {tesa_path} {out_dir}")
os.system(f"cp {model_path} {out_dir}")
os.system(f"cp {state_path} {out_dir}")
os.system(f"cp {id_map_path} {out_dir}")