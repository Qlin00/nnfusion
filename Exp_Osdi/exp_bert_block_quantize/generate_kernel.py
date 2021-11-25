import os
import sys
import json
import torch
from numpy import core
import onnx
import re
from copy import deepcopy
import numpy as np
import math
from SparGen.Common.Utils import *
# need to replace for the right quantized values
default_kv = {}
default_kv["CHUNK_K_VALUE"] = 8
default_kv["BLOCK_ROW_WARPS_VALUE"] = 1
default_kv["BLOCK_COL_WARPS_VALUE"] = 2
default_kv["WARP_ROW_TILES_VALUE"] = 2
default_kv["WARP_COL_TILES_VALUE"] = 1
tune_kernel_cfg = {}
if os.path.exists('tuning_cfg.json'):
    with open('tuning_cfg.json', 'r') as f:
        tune_kernel_cfg = json.load(f)

launch_cfg = {}
tesa = torch.load('./tesa')
for tesaid in tesa:
    # if tesaid ==5:
    #     import pdb; pdb.set_trace()
    launch_cfg[tesaid] = deepcopy(default_kv)
    if tesa[tesaid]['weight'].size(1) < 128:
        launch_cfg[tesaid]['CHUNK_K_VALUE'] = 4
    if str(tesaid) in tune_kernel_cfg:
        launch_cfg[tesaid].update(tune_kernel_cfg[str(tesaid)])


sparse_block = {}
for key in launch_cfg:
    sparse_block[int(key)] =  ( launch_cfg[key]["WARP_ROW_TILES_VALUE"] * launch_cfg[key]["BLOCK_ROW_WARPS_VALUE"] *16, launch_cfg[key]['CHUNK_K_VALUE']* 16) 

generate_block_quantize_cfg('./tesa', 'nni_weight.pth', './tesaid_2_names', 'nnfusion_cfg', sparse_block_cfg=sparse_block)
if os.path.exists('/home/v-linbin/.cache/nnfusion/kernel_cache.db'):
    os.remove('/home/v-linbin/.cache/nnfusion/kernel_cache.db')

prefix = 'kernel'
os.makedirs(prefix, exist_ok=True)


with open('../Template/block_quantize_template_bias.json', 'r') as f:
    template = json.load(f)
    template = template[0]

with open('../Template/block_quantize_template_bias.cu', 'r') as f:
    code = f.read()

with open('../Template/quantize_dot_template_bias.json', 'r') as f:
    dense_template = json.load(f)
    dense_template = dense_template[0]

with open('../Template/quantize_dot_template_bias.cu', 'r') as f:
    dense_code = f.read()


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
        out_shape = shape_info[torch_name]['out_shape'][0]
        sparse_type = line[1]
        if sparse_type == "BlockQuantize":
            kv = {}
            kv["M_VALUE"] = np.prod(in_shape[:-1])
            kv["K_VALUE"] = in_shape[-1]
            kv["N_VALUE"] = weight_shape[0]
            kv.update(launch_cfg[tesa_id])
            kv['COMMENT_TAG'] = f"TESAID : {tesa_id}"

            print(in_shape)
            print(weight_shape)
            assert in_shape[-1] == weight_shape[1]
            new_code = code
            for k, v in launch_cfg[tesa_id].items():
                # print(k, v)
                new_code = new_code.replace(k, str(v))
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
        elif sparse_type=="Quantize":
            pass
            kv = {}
            m = np.prod(in_shape[:-1])
            k = in_shape[-1]
            n = weight_shape[0]
            kv["M_GLOBAL_VALUE"] = np.prod(in_shape[:-1])
            kv["K_GLOBAL_VALUE"] = in_shape[-1]
            kv["N_GLOBAL_VALUE"] = weight_shape[0]
            kv.update(launch_cfg[tesa_id])
            kv['COMMENT_TAG'] = f"TESAID : {tesa_id}"

            print(in_shape)
            print(weight_shape)
            assert in_shape[-1] == weight_shape[1]
            new_code = dense_code
            for k, v in launch_cfg[tesa_id].items():
                # print(k, v)
                new_code = new_code.replace(k, str(v))
            for k, v in kv.items():
                new_code = new_code.replace(k, str(v))
            
            # import pdb; pdb.set_trace()
            dense_template['code'] = new_code + tesa_id * ' '
            dense_template['kernel_identifier'] = kernel_id
            dense_template['op_type'] = 'QuantizeDotAdd'
            block_size_m = 16 * kv['BLOCK_ROW_WARPS_VALUE'] * kv["WARP_ROW_TILES_VALUE"]
            block_size_n = 16 * kv["BLOCK_COL_WARPS_VALUE"] * kv["WARP_COL_TILES_VALUE"]
            block_size_k = 16 * kv["CHUNK_K_VALUE"]
            
            Block_num = math.ceil((m * n) / (block_size_m * block_size_n))
            warp_size =32
            Thread_per_block = (warp_size * kv["BLOCK_ROW_WARPS_VALUE"] * kv["BLOCK_COL_WARPS_VALUE"])
            dense_template['gridDim'] = [Block_num,1,1]
            dense_template['blockDim'] = [Thread_per_block,1,1]
            dense_template['parameters']['out_shape'] = out_shape

            f_path =  os.path.join(prefix, f"{tesa_id}.json")
            print(f_path)
            with open(f_path, 'w') as f:
                json.dump(dense_template, f)
            os.system(f"python ../../src/tools/nnfusion/kernel_db/convert_external_spargen.py {f_path}")