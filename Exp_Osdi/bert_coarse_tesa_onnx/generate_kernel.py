import os
import sys
import json
import torch
from numpy import core
import onnx
import re
import numpy as np

prefix = 'kernel'
os.makedirs(prefix, exist_ok=True)
with open('../Template/block_sparse_template_bias.json', 'r') as f:
    template = json.load(f)
    template = template[0]

with open('../Template/block_sparse_template_bias.cu', 'r') as f:
    code = f.read()

with open('bert_coarse_pruned_shape.json', 'r') as f:
    shape_info = json.load(f)

id2name = torch.load('tesaid_2_names')

config = {}

with open('nnfusion_cfg/config', 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = re.split(' ', line)
        tesa_id = int(line[0])
        kernel_id = line[2]
        torch_name = id2name[tesa_id][0]
        in_shape = shape_info[torch_name]['in_shape'][0]
        weight_shape = shape_info[torch_name]['weight_shape'][0]
        kv = {}
        kv['BLOCK_SIZE_M_VALUE'] = 32
        kv['BLOCK_SIZE_K_VALUE'] = 32
        kv['BLOCK_SIZE_N_VALUE'] = 32
        kv['THREAD_SIZE_M_VALUE'] = 8
        kv['THREAD_SIZE_K_VALUE'] = 4
        kv['THREAD_SIZE_N_VALUE'] = 8
        kv["M_VALUE"] = np.prod(in_shape[:-1])
        kv["K_VALUE"] = in_shape[-1]
        kv["N_VALUE"] = weight_shape[0]


        print(in_shape)
        print(weight_shape)
        assert in_shape[-1] == weight_shape[1]
        new_code = code
        for k, v in kv.items():
            new_code = new_code.replace(k, str(v))
        # import pdb; pdb.set_trace()
        template['code'] = new_code + tesa_id * ' '
        template['kernel_identifier'] = kernel_id
        template['op_type'] = 'SparseDot'
        grid_dim = [int(kv['M_VALUE']/kv["BLOCK_SIZE_M_VALUE"]), int(kv['N_VALUE']/kv["BLOCK_SIZE_N_VALUE"]), 1]
        template['gridDim'] = grid_dim
        block_dim = [int(kv['BLOCK_SIZE_M_VALUE']/kv['THREAD_SIZE_M_VALUE']), int(kv['BLOCK_SIZE_N_VALUE']/kv['THREAD_SIZE_N_VALUE']), 1]
        template['blockDim'] = block_dim
        f_path =  os.path.join(prefix, f"{tesa_id}.json")
        print(f_path)
        with open(f_path, 'w') as f:
            json.dump(template, f)
        os.system(f"python ../../src/tools/nnfusion/kernel_db/convert_external_spargen.py {f_path}")