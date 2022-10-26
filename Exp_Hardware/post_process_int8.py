from ast import arg
import re
import os
import argparse

prefix = "#include <mma.h>\nusing namespace nvcuda;\n"
parser = argparse.ArgumentParser()
parser.add_argument('--file', help='file path')
args = parser.parse_args()
code_path = args.file

with open(code_path, 'r') as f:
    code = f.read()

code = prefix + code
pattern = "QuantizeDot_float_float_float_float_float_float_float_float_cuda_QuantizeDot_[0-9]+<<<"
func_calls = re.findall(pattern, code)
# import ipdb; ipdb.set_trace()
for f in func_calls:
    # 65536 
    _tmp = 'cudaFuncSetAttribute({}, cudaFuncAttributeMaxDynamicSharedMemorySize,73728);\n{}'.format(f[:-3], f)
    code = code.replace(f, _tmp)
with open(code_path, 'w') as f:
    f.write(code)