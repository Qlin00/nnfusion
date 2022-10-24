import json
import os
import torch
prefix = 'balance_bert_large_n_4_m_32_align1'
_path = os.path.join(prefix, 'balance_interface.pth')
data = torch.load(_path)
shape_set = set()
for key in data:
    shape_set.add((data[key]['M'], data[key]['K'], data[key]['N']))

import ipdb; ipdb.set_trace()
print(shape_set)