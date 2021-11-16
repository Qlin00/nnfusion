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
default_kv["BLOCK_SIZE_M_VALUE"] = 32
default_kv["BLOCK_SIZE_K_VALUE"] = 8
default_kv["BLOCK_SIZE_N_VALUE"] = 128
default_kv["THREAD_SIZE_M_VALUE"] = 8
default_kv["THREAD_SIZE_K_VALUE"] = 4
default_kv["THREAD_SIZE_N_VALUE"] = 8
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
    if str(tesaid) in tune_kernel_cfg:
        launch_cfg[tesaid].update(tune_kernel_cfg[str(tesaid)])


generate_quantize_dot_cfg('./tesa', 'nni_weight.pth', './tesaid_2_names', 'nnfusion_cfg')
if os.path.exists('/home/v-linbin/.cache/nnfusion/kernel_cache.db'):
    os.remove('/home/v-linbin/.cache/nnfusion/kernel_cache.db')

prefix = 'kernel'
os.makedirs(prefix, exist_ok=True)


with open('../Template/rocm_quantize_dot_template_bias.json', 'r') as f:
    template = json.load(f)
    template = template[0]

with open('../Template/rocm_quantize_dot_template_bias.cu', 'r') as f:
    code = f.read()


with open('hubert_coarse_shape.json', 'r') as f:
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
        print(torch_name)
        if 'conv' in torch_name:
            continue
        in_shape = shape_info[torch_name]['in_shape'][0]
        weight_shape = shape_info[torch_name]['weight_shape'][0]
        out_shape = shape_info[torch_name]['out_shape'][0]
        kv = {}
        m = np.prod(in_shape[:-1])
        k = in_shape[-1]
        n = weight_shape[0]
        kv["GLOBAL_M_VALUE"] = np.prod(in_shape[:-1])
        kv["GLOBAL_K_VALUE"] = in_shape[-1]
        kv["GLOBAL_N_VALUE"] = weight_shape[0]
        kv.update(launch_cfg[tesa_id])
        kv['COMMENT_TAG'] = f"TESAID : {tesa_id}"
        print(torch_name)
        print(kv["GLOBAL_M_VALUE"], kv["BLOCK_SIZE_M_VALUE"])
        
        if kv["GLOBAL_M_VALUE"] % kv["BLOCK_SIZE_M_VALUE"] != 0:
            import pdb; pdb.set_trace()
            continue
        print(kv["GLOBAL_N_VALUE"], kv["BLOCK_SIZE_N_VALUE"])

        if kv["GLOBAL_N_VALUE"] % kv["BLOCK_SIZE_N_VALUE"] != 0:
            continue
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
        template['op_type'] = 'QuantizeDotAdd'

      
        template['gridDim'] = [int(kv['GLOBAL_N_VALUE'] / kv['BLOCK_SIZE_N_VALUE']), int(kv['GLOBAL_M_VALUE'] / kv['BLOCK_SIZE_M_VALUE']), 1]
        template['blockDim'] = [int(kv['BLOCK_SIZE_N_VALUE']/kv['THREAD_SIZE_N_VALUE']), int(kv['BLOCK_SIZE_M_VALUE']/kv['THREAD_SIZE_M_VALUE']), 1]
        template['parameters']['out_shape'] = out_shape

        f_path =  os.path.join(prefix, f"{tesa_id}.json")
        print(f_path)
        with open(f_path, 'w') as f:
            json.dump(template, f)
        os.system(f"python ../../src/tools/nnfusion/kernel_db/convert_external_spargen.py {f_path} ROCM_GPU")