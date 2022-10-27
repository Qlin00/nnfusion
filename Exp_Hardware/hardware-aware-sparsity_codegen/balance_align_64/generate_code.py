import torch
import json
import argparse

parser = argparse.ArgumentParser(description='Process interface file.')
parser.add_argument( '--file', action = 'store', type = str, default="algorithm_kernel_interface.pth", help = 'The text to parse.' )

args = parser.parse_args()

filename = args.file
algorithm_kernel_interface = torch.load(filename)
template = "int8_small_shared.cu"
template = "int8_large_shared.cu"
f = open(template)
template_content = f.read()
kernel_dict = {}

for name, kernel_content in algorithm_kernel_interface.items():
    template_tmp = template_content
    M, K, N = kernel_content['M'], kernel_content['K'], kernel_content['N']
    align_n = kernel_content['align_n']
    sparsity = kernel_content['sparsity']
    template_config = {}

    template_config['M_GLOBAL_VALUE'] = M
    template_config['N_GLOBAL_VALUE'] = N
    template_config['K_GLOBAL_VALUE'] = K
    template_config['SPARSITY_VALUE'] = sparsity
    if sparsity == 0.5:
        template_config['CHUNK_K_VALUE'] = 32
    else:
        template_config['CHUNK_K_VALUE'] = 64
        
    for key, value in template_config.items():
        template_tmp=template_tmp.replace(key, str(value))

    # get the launch config
    block_row_warps = 2
    block_col_warps = 4
    warp_row_tiles = 4
    warp_col_tiles = 2
    block_col_tiles = warp_col_tiles * block_col_warps
    block_row_tiles = warp_row_tiles * block_row_warps
    block_size_col = 16 * block_col_tiles
    block_size_row = 16 * block_row_tiles
    block_num = M * N // (block_size_row * block_size_col)
    warps_per_block = block_row_warps * block_col_warps
    threads_per_block = 32 * warps_per_block

    launch_config = {}
    launch_config['dimGrid'] = [block_num, 1]
    launch_config['dimBlock'] = [threads_per_block, 1]

    kernel_dict[name] = {'code': template_tmp, 'launch_config': launch_config}

    print(f"name: {name}, m: {M}, k: {K}, n: {N}, sparsity: {sparsity}")

with open("kernel_dict.json", "w") as outfile:
    json.dump(kernel_dict, outfile)