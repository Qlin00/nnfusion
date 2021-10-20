import os
import sys
import json
import torch
from numpy import core
import onnx
import re
import numpy as np
from SparGen.Common.Utils import *
# need to replace for the right quantized values
generate_block_quantize_cfg('./tesa', 'nni_weight.pth', './tesaid_2_names', 'nnfusion_cfg', block_h=16, block_w=16)

if os.path.exists('/home/v-linbin/.cache/nnfusion/kernel_cache.db'):
    os.remove('/home/v-linbin/.cache/nnfusion/kernel_cache.db')
prefix = 'kernel'
os.makedirs(prefix, exist_ok=True)


with open('../Template/block_quantize_template_bias.json', 'r') as f:
    template = json.load(f)
    template = template[0]

with open('../Template/block_quantize_template_bias.cu', 'r') as f:
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
        kv["CHUNK_K_VALUE"] = 1
        kv["M_VALUE"] = np.prod(in_shape[:-1])
        kv["K_VALUE"] = in_shape[-1]
        kv["N_VALUE"] = weight_shape[0]
        kv["BLOCK_ROW_WARPS_VALUE"] = 1
        kv["BLOCK_COL_WARPS_VALUE"] = 2
        kv["WARP_ROW_TILES_VALUE"] = 2
        kv["WARP_COL_TILES_VALUE"] = 1

        print(in_shape)
        print(weight_shape)
        assert in_shape[-1] == weight_shape[1]
        new_code = code
        for k, v in kv.items():
            new_code = new_code.replace(k, str(v))
        # import pdb; pdb.set_trace()
        template['code'] = new_code + tesa_id * ' '
        template['kernel_identifier'] = kernel_id
        template['op_type'] = 'BlockQuantizeDotAdd'
        block_size_m = 16 * kv["BLOCK_ROW_WARPS_VALUE"] * kv["WARP_ROW_TILES_VALUE"]
        block_size_n = 16 * kv["BLOCK_COL_WARPS_VALUE"] * kv["WARP_COL_TILES_VALUE"]
        grid_dim = [int(kv["M_VALUE"]*kv['N_VALUE']/block_size_m/block_size_n), 1, 1]
        template['gridDim'] = grid_dim
        thread_per_block = (32 * kv["BLOCK_ROW_WARPS_VALUE"] * kv["BLOCK_COL_WARPS_VALUE"])

 
        # block_dim = [int(kv['BLOCK_SIZE_N_VALUE']/kv['THREAD_SIZE_N_VALUE']), int(kv['BLOCK_SIZE_M_VALUE']/kv['THREAD_SIZE_M_VALUE']), 1]
        template['blockDim'] = [thread_per_block, 1, 1]
        f_path =  os.path.join(prefix, f"{tesa_id}.json")
        print(f_path)
        with open(f_path, 'w') as f:
            json.dump(template, f)
        os.system(f"python ../../src/tools/nnfusion/kernel_db/convert_external_spargen.py {f_path}")