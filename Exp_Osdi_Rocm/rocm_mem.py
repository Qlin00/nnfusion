import re
import os
import sys
import time
from time import sleep

time_interval = 2
device = int(sys.argv[1])

while True:
    sleep(1.0 * time_interval/1000)
    # for i in range(1000):
    #    pass
    os.system(f"rocm-smi --showmeminfo vram | grep \"GPU\[{device}\]\" | grep Used")
