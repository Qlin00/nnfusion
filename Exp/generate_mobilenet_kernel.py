import json
import os
import sys
import math
prefix = 'mobile_kernel_batch8_coarse0.6_quantize'
# with open('Mobilenet_shape.json', 'r') as jf:
#     data = json.load(jf)
with open('Mobilenet_coarse0.6_shape_batch8.json', 'r') as jf:
    data = json.load(jf)
os.makedirs(prefix, exist_ok=True)
djson = {}
dcu = None    
with open('./kernel/conv.json.template', 'r') as jf:
    djson = json.load(jf)
with open('./kernel/depthwise.template', 'r') as cuf:
    dcu = cuf.read()

for conv in data:
    weight_shape = data[conv]['weight_shape'][0]
    out_shape = data[conv]['out_shape'][0]
    in_shape = data[conv]['in_shape'][0]
    padding = data[conv]['padding']
    stride = data[conv]['stride']
    groups = data[conv]['groups']
    if groups!=in_shape[1]:
        continue
    out_size = 1
    for i in out_shape:
        out_size *= i
    print(out_size)


    dilation = data[conv]['dilation']
    djson[0]['parameters']['input_shape'] = in_shape
    djson[0]['parameters']['output_shape'] = out_shape
    djson[0]['parameters']['filter_shape'] = weight_shape
    djson[0]['parameters']['window_movement_strides'] = stride
    djson[0]['parameters']['window_dilation_strides'] = dilation
    djson[0]['parameters']['padding_below_diff'] = padding
    djson[0]['op_type'] = 'QuantizeDepthwiseConv2dNative'
    code = dcu
    code = code.replace('NTHREADS', str(out_size))
    code = code.replace('CHANNELS', str(in_shape[1]))
    code = code.replace('INPUT_HEIGHT', str(in_shape[2]))
    code = code.replace('INPUT_WIDTH', str(in_shape[3]))
    code = code.replace('OUTPUT_HEIGHT', str(out_shape[2]))
    code = code.replace('OUTPUT_WIDTH', str(out_shape[3]))
    code = code.replace('KERNEL_H', str(weight_shape[2]))
    code = code.replace('KERNEL_W', str(weight_shape[3]))
    code = code.replace('STRIDE_H', str(stride[0]))
    code = code.replace('STRIDE_W', str(stride[1]))
    code = code.replace('PAD_H', str(padding[0]))
    code = code.replace('PAD_W', str(padding[1]))
    djson[0]["code"] = code
    print(weight_shape)

    djson[0]['gridDim'][0]=int(math.ceil(out_size/512.0))
    djson[0]['blockDim'][0]=512
    file_name = 'mobilenet' + conv.replace('.', '_')
    filepath = os.path.join(prefix, file_name)
    with open(filepath+'.json', 'w') as jf:
        json.dump(djson, jf) 
    with open(filepath+'.cu', 'w') as f:
        f.write(code)  

with open('./kernel/dot_add_relu.template', 'r') as cuf:
    dcu = cuf.read()

for conv in data:
    weight_shape = data[conv]['weight_shape'][0]
    if(weight_shape[3]!=1):
        continue
    out_shape = data[conv]['out_shape'][0]
    in_shape = data[conv]['in_shape'][0]
    padding = data[conv]['padding']
    stride = data[conv]['stride']
    dilation = data[conv]['dilation']
    djson[0]['parameters']['input_shape'] = in_shape
    djson[0]['parameters']['output_shape'] = out_shape
    djson[0]['parameters']['filter_shape'] = weight_shape
    djson[0]['parameters']['window_movement_strides'] = stride
    djson[0]['parameters']['window_dilation_strides'] = dilation
    djson[0]['parameters']['padding_below_diff'] = padding
    djson[0]['op_type'] = 'QuantizeConvolution'
    code = dcu
    M = in_shape[0]*in_shape[2]*in_shape[3] # N * H * W
    K = in_shape[1]
    N = out_shape[1]
    # if(M%16!=0):
    #     continue
    print(conv)
    print(M,K,N)
    # assert M%16 ==0
    assert K%16 == 0
    assert N%16==0
    code=code.replace('MGLOBAL_VALUE', str(M))
    code=code.replace('KGLOBAL_VALUE', str(K))
    code=code.replace('NGLOBAL_VALUE', str(N))
    code=code.replace('MTILES_VALUE', str(int(M/16)))
    code=code.replace('KTILES_VALUE', str(int(K/16)))
    code=code.replace('NTILES_VALUE', str(int(N/16)))
    
    djson[0]["code"] = code
    print(weight_shape)


    djson[0]['gridDim'][0]=68
    djson[0]['blockDim'][0] = 256
    file_name = 'conv1x1_'+str(M)+'_'+str(K)+'_'+str(N)
    filepath = os.path.join(prefix, file_name)
    with open(filepath+'.json', 'w') as jf:
        json.dump(djson, jf) 
    with open(filepath+'.cu', 'w') as f:
        f.write(code)  