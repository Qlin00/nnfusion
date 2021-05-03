import json
import sys
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--all', default=False, action='store_true')
parser.add_argument('--prefix', default=None)
parser.add_argument('--json', default=None, type=str)
parser.add_argument('--cu', default=None, type=str)
args = parser.parse_args()


def inject(jsonf, cuf):

    with open(jsonf, "r") as f:
        data = json.load(f)
    with open(cuf, "r") as f:
        code = f.read()

    data[0]["code"] = code
    # print(data[0]["code"])
    with open(jsonf, "w") as f:
        json.dump(data, f, indent=4)

if args.all:
    PREFIX = args.prefix
    files = os.listdir(PREFIX)

    for f in files:
        name = os.path.splitext(f)[0]
        suffix = os.path.splitext(f)[-1]
        if os.path.splitext(f)[-1]=='.json':
            jsonf = os.path.join(PREFIX, f)
            cuf = os.path.join(PREFIX, name+'.cu')
            assert os.path.exists(jsonf)
            assert os.path.exists(cuf)

            # print(jsonf, cuf)
            inject(jsonf, cuf)
            os.system("python ../src/tools/nnfusion/kernel_db/convert_external_quantize.py %s"%jsonf)
else:
    inject(args.json, args.cu)
