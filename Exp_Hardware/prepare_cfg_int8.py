# from hardware_sparse.SparTA.sparta.common.utils import inject_kernel_with_id
from sparta.common.utils import generate_balance_cfg_int8, inject_kernel, inject_kernel_with_id
import argparse
import os
import json
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', help='input dir')
    parser.add_argument('--out_dir', help='output_dir')
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    tesa_path = os.path.join(args.in_dir, 'tesa')
    state_path = os.path.join(args.in_dir, 'state_dict.pth')
    id_map_path = os.path.join(args.in_dir, 'tesaid_2_names')
    id_map = os.path.join(args.in_dir, 'tesaid_2_names')
    sparse_block_cfg = {}
    with open('kernel_dict.json', 'r') as f:
        kernels = json.load(f)

    id_maps = torch.load(id_map_path)
    name_2_tid= {}
    id_2_name = {}
    for tid, names in id_maps.items():
        id_2_name[tid] = names[0]
        name_2_tid[names[0]] =tid

    onnx_path = os.path.join(args.in_dir, 'model_tesa.onnx')
    os.system('cp {} {}'.format(onnx_path, args.out_dir))
    kernel_path = 'kernel_dict.json'
    template_path = 'block_sparse_template_bias_row_int8.json'

    
    inject_kernel_with_id(template_path, kernel_path, 'BlockQuantizeDotAdd', id_map_path, os.path.join(args.out_dir, 'kernel'))
    # generate_balance_cfg_int8(args.out_dir, 64, 32, 0.5)