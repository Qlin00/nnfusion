from ast import arg
import re
import os
import argparse

prefix = "#include <mma.h>\nusing namespace nvcuda;\n"
parser = argparse.ArgumentParser()
parser.add_argument('--file', help='file path')
args = parser.parse_args()
code_path = args.file

