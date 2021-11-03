import os
import sys
import json
from tempfile import template
import torch
from numpy import core
import json

import onnx
import re
import numpy as np
from SparGen.Common.Utils import *

if os.path.exists('/home/v-linbin/.cache/nnfusion/kernel_cache.db'):
    os.remove('/home/v-linbin/.cache/nnfusion/kernel_cache.db')

tesa_path = './tesa'
state_path = './bert_finegrained_0.95_nomask.pth'
id_2_maps = './tesaid_2_names'
out_dir = './nnfusion_cfg'
prefix = 'kernel'
shape_path = 'bert_ori_shape.json'

generate_sputnik_sparse_cfg(tesa_path, state_path, id_2_maps, out_dir)

# with open('../Template/sputnik_sparse_template.cu') as f:
#     code = f.read()
# with open('../Template/sputnik_sparse_template.json') as f:
#     template = json.load(f)
#     template = template[0]

# with open(shape_path, 'r') as f:
#     shape_info = json.load(f)


# id2name = torch.load('tesaid_2_names')
# with open('nnfusion_cfg/config', 'r') as f:
#     lines = f.readlines()
#     for line in lines:
#         line = re.split(' ', line)
#         tesa_id = int(line[0])
#         kernel_id = line[2]
#         torch_name = id2name[tesa_id][0]
#         in_shape = shape_info[torch_name]['in_shape'][0]
#         weight_shape = shape_info[torch_name]['weight_shape'][0]
#         m = weight_shape[0]
#         k = weight_shape[1]
#         n = 1
#         for i in range(len(in_shape)-1):
#             n*=in_shape[i]
#         kv = {"M_VALUE":m, "K_VALUE":k, "N_VALUE":n}
#         for k, v in kv.items():
#             new_code = new_code.replace(k, str(v))
#         new_code = code + ' ' * tesa_id
#         template['code'] = new_code
#         # import pdb; pdb.set_trace()
#         template['kernel_identifier'] = kernel_id
#         template['op_type'] = 'SputnikDot'

#         f_path = os.path.join(prefix, f"{tesa_id}.json")
#         print(f_path)
#         with open(f_path, 'w') as f:
#             json.dump(template, f)
#         os.system(f"python ../../src/tools/nnfusion/kernel_db/convert_external_spargen.py {f_path}")