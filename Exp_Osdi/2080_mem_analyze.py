import re
import sys
import os
log_path = sys.argv[1]
if not os.path.exists(log_path):
    print(f"{log_path} does not exists")
peak = 0
with open(log_path, 'r') as f:
    lines = f.readlines()
    for line in lines[1:]:
        if 'MiB' in line:
            peak = max(peak, int(re.split(' ',line)[0]))

print('Peak Memory Usage', peak)