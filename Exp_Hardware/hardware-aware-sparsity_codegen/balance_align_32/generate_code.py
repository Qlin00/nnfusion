import torch
import json
import argparse

parser = argparse.ArgumentParser(description='Process interface file.')
parser.add_argument( '--file', action = 'store', type = str, default="algorithm_kernel_interface.pth", help = 'The text to parse.' )

args = parser.parse_args()

filename = args.file
algorithm_kernel_interface = torch.load(filename)
template = "balance_align_shared_32.cu"
f = open(template)
template_content = f.read()
kernel_dict = {}

for name, kernel_content in algorithm_kernel_interface.items():
    template_tmp = template_content
    M, K, N = kernel_content['M'], kernel_content['K'], kernel_content['N']
    align_n = kernel_content['align_n']
    sparsity = kernel_content['sparsity']
    template_config = {}

    template_config['BLOCK_SIZE_M_VALUE'] = 32
    template_config['BLOCK_SIZE_N_VALUE'] = 32
    if sparsity == 0.5:
        template_config['BLOCK_SIZE_K_VALUE'] = 128
    else:
        template_config['BLOCK_SIZE_K_VALUE'] = 256
    template_config['THREAD_SIZE_M_VALUE'] = 4
    template_config['THREAD_SIZE_N_VALUE'] = 4

    template_config['BANK_VAL_VALUE'] = 32
    template_config['M_GLOBAL_VALUE'] = M
    template_config['N_GLOBAL_VALUE'] = N
    template_config['K_GLOBAL_VALUE'] = K
    template_config['SPARSITY_VALUE'] = sparsity
    for key, value in template_config.items():
        template_tmp=template_tmp.replace(key, str(value))

    # get the launch config
    block_size_m = template_config['BLOCK_SIZE_M_VALUE']
    block_size_k = template_config['BLOCK_SIZE_K_VALUE']
    block_size_n = template_config['BLOCK_SIZE_N_VALUE']
    thread_size_m = template_config['THREAD_SIZE_M_VALUE']
    thread_size_n = template_config['THREAD_SIZE_N_VALUE']
    launch_config = {}
    launch_config['dimBlock'] = [int((block_size_m/thread_size_m)*(block_size_n/thread_size_n)), 1]
    launch_config['dimGrid'] = [int(M / block_size_m), int(N / block_size_n)]

    kernel_dict[name] = {'code': template_tmp, 'launch_config': launch_config}

    print(f"name: {name}, m: {M}, k: {K}, n: {N}, sparsity: {sparsity}")

with open("kernel_dict.json", "w") as outfile:
    json.dump(kernel_dict, outfile)