import os
import sys
import re
total = 16368.0
log_path = sys.argv[1]
if not os.path.exists(log_path):
    raise Exception(f"{log_path} not exists")

peak = 0
with open(log_path, 'r') as f:
    lines = f.readlines()
    for line in lines:
        m_ratio = float(re.split(' ', line)[-1])
        peak = max(peak, m_ratio)

print('Peak Memory', peak*total/100)