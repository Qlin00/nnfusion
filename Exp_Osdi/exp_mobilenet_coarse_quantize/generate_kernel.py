import os
import sys
import json
from types import CodeType
import torch
from numpy import core
import math
import onnx
from copy import deepcopy
import re
import numpy as np
from SparGen.Common.Utils import *
from utils import *
from shape_hook import ShapeHook
from nni.compression.pytorch.speedup import ModelSpeedup
from nni.compression.pytorch.utils import get_module_by_name
tune_kernel_cfg = {}
sparse_block = {}
if os.path.exists('tuning_cfg.json'):
    with open('tuning_cfg.json', 'r') as f:
        tune_kernel_cfg = json.load(f)

if os.path.exists('/home/v-linbin/.cache/nnfusion/kernel_cache.db'):
    os.remove('/home/v-linbin/.cache/nnfusion/kernel_cache.db')
prefix = 'kernel'
os.makedirs(prefix, exist_ok=True)
dummy_input = torch.rand(32,3,224,224)
model = create_model('mobilenet_v1')
# ms = ModelSpeedup(model, dummy_input, 'mask_temp.pth')
# ms.speedup_model()
align_speedup(model, dummy_input, 'mask_temp.pth')
model.load_state_dict(torch.load('finetune_weights.pth'))
sh = ShapeHook(deepcopy(model), dummy_input)
sh.export('mobilenet_coarse_shape.json')
tmp_model = deepcopy(model)
export_tesa(tmp_model, dummy_input, 'nnfusion_cfg')

generate_mobilenet_quantize_cfg('./nnfusion_cfg/tesa', 'finetune_weights.pth', './nnfusion_cfg/tesaid_2_names', 'nnfusion_cfg')

with open('mobilenet_coarse_shape.json', 'r') as f:
    shape_info = json.load(f)



with open('../Template/quantize_dot_template_bias.json', 'r') as f:
    template = json.load(f)
    conv1x1_template = template[0]

with open('../Template/quantize_dot_template_bias.cu', 'r') as f:
    conv1x1_code = f.read()

with open('../Template/depth_quantize_temlate_bias.json', 'r') as f:
    template = json.load(f)
    depth_template = template[0]

with open('../Template/depth_quantize_temlate_bias.cu', 'r') as f:
    depth_code = f.read()



tesa = torch.load('nnfusion_cfg/tesa')
id2name = torch.load('nnfusion_cfg/tesaid_2_names')

config = {}
def generate_conv1x1(module):
    pass
def generate_depth(module):
    pass
with open('nnfusion_cfg/config', 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = re.split(' ', line)
        tesa_id = int(line[0])
        kernel_id = line[2]
        torch_name = id2name[tesa_id][0]
        in_shape = shape_info[torch_name]['in_shape'][0]
        out_shape = shape_info[torch_name]['out_shape'][0]
        weight_shape = shape_info[torch_name]['weight_shape'][0]
        _, module = get_module_by_name(model, torch_name)
        f_path =  os.path.join(prefix, f"{tesa_id}.json")

        kv = {}
        if isinstance(module, torch.nn.Conv2d):
            print(torch_name, module.weight.size())
            if(module.groups>1):
                # depthwise conv
                kv["CHANNEL_VALUE"]=module.in_channels
                kv["IN_HEIGHT_VALUE"]= in_shape[2]
                kv["IN_WIDTH_VALUE"] = in_shape[3]
                kv["OUT_HEIGHT_VALUE"] = out_shape[2]
                kv["OUT_WIDTH_VALUE"] = out_shape[3]
                kv["BATCHSIZE_VALUE"] = 32
                kv["KERNEL_H_VALUE"] = module.kernel_size[0]
                kv["KERNEL_W_VALUE"] = module.kernel_size[1]
                kv["STRIDE_H_VALUE"] = module.stride[0]
                kv["STRIDE_W_VALUE"] = module.stride[1]
                kv["PAD_H_VALUE"] = module.padding[0]
                kv["PAD_W_VALUE"] = module.padding[1]
                code = depth_code
                for k, v in kv.items():
                    code = code.replace(k, str(v))
                depth_template['code'] = code + ' '*tesa_id
                depth_template['kernel_identifier'] = kernel_id
                grid_dim = [math.ceil( out_shape[0] * out_shape[1]* out_shape[2] * out_shape[3] / 512), 1,  1]
                depth_template['gridDim'] = grid_dim
                block_dim = [512, 1, 1]
                depth_template['blockDim'] = block_dim
                depth_template['op_type'] = 'QuantizeDepthwiseConv2dNative'
                with open(f_path, 'w') as f:
                    json.dump(depth_template, f)
            elif weight_shape[2] == 1:
                # conv 1x1
                m = in_shape[0] * in_shape[2] * in_shape[3]
                k = in_shape[1]
                n = out_shape[1]

                kv["M_GLOBAL_VALUE"] = m
                kv["K_GLOBAL_VALUE"] = k
                kv["N_GLOBAL_VALUE"] = n
                kv["CHUNK_K_VALUE"] = 1
                kv["BLOCK_ROW_WARPS_VALUE"] = 1
                kv["BLOCK_COL_WARPS_VALUE"] = 2
                kv["WARP_ROW_TILES_VALUE"] = 2
                kv["WARP_COL_TILES_VALUE"] = 1
                block_size_m = 16 * kv['BLOCK_ROW_WARPS_VALUE'] * kv["WARP_ROW_TILES_VALUE"]
                block_size_n = 16 * kv["BLOCK_COL_WARPS_VALUE"] * kv["WARP_COL_TILES_VALUE"]
                block_size_k = 16 * kv["CHUNK_K_VALUE"]
                Block_num = math.ceil((m * n) / (block_size_m * block_size_n))
                warp_size =32
                Thread_per_block = (warp_size * kv["BLOCK_ROW_WARPS_VALUE"] * kv["BLOCK_COL_WARPS_VALUE"])
                
                code = conv1x1_code
                for k, v in kv.items():
                    code = code.replace(k, str(v))
                conv1x1_template['op_type'] = 'QuantizeConvolution'
                conv1x1_template['code'] = code +  ' '* tesa_id
                conv1x1_template['kernel_identifier'] = kernel_id
                conv1x1_template['gridDim'] = [Block_num,1,1]
                conv1x1_template['blockDim'] = [Thread_per_block,1,1]
                with open(f_path, 'w') as f:
                    json.dump(conv1x1_template, f)
            else:
                continue
        elif isinstance(module, torch.nn.Linear):
            # TODO also replace the final linear
            continue
        else:
            continue 
        # import pdb; pdb.set_trace()
        # if os.path.exists(f_path):
        os.system(f"python ../../src/tools/nnfusion/kernel_db/convert_external_spargen.py {f_path}")