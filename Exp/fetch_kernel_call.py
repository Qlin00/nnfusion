import re
import os
import sys
print(sys.argv[1])
assert os.path.exists(sys.argv[1])
with open(sys.argv[1], 'r') as f:
    data = f.read()

pattern = 'QuantizeDot.*QuantizeDot.*<<<'

pattern = re.compile(pattern)
result = pattern.findall(data)
print(result)
for kernel in result:
    print('cudaFuncSetAttribute(%s , cudaFuncAttributeMaxDynamicSharedMemorySize,65536);' % kernel[:-3])


pattern = 'QuantizeConv1x1.*QuantizeConv1x1.*<<<'

pattern = re.compile(pattern)
result = pattern.findall(data)
print(result)
for kernel in result:
    print('cudaFuncSetAttribute(%s , cudaFuncAttributeMaxDynamicSharedMemorySize,65536);' % kernel[:-3])


pattern = 'QuantizeDepthwiseConv2dNative.*QuantizeDepthwiseConv2dNative.*<<<'

pattern = re.compile(pattern)
result = pattern.findall(data)
print(result)
for kernel in result:
    print('cudaFuncSetAttribute(%s , cudaFuncAttributeMaxDynamicSharedMemorySize,65536);' % kernel[:-3])
