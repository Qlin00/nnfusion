import json
import os
import sys
prefix = 'efficientb1_kernel'
with open('Efficientb1_shape.json', 'r') as jf:
    data = json.load(jf)
os.makedirs(prefix, exist_ok=True)
djson = {}
dcu = None    
with open('./kernel/conv.json.template', 'r') as jf:
    djson = json.load(jf)
with open('./kernel/depthwise.template', 'r') as cuf:
    dcu = cuf.read()
global_suffix = ''
for conv in data:
    for suffix in ["", "BatchNormInference", "BatchNormInferenceSigmoid"]:
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
        assert(out_size%512==0)

        dilation = data[conv]['dilation']
        djson[0]['parameters']['input_shape'] = in_shape
        djson[0]['parameters']['output_shape'] = out_shape
        djson[0]['parameters']['filter_shape'] = weight_shape
        djson[0]['parameters']['window_movement_strides'] = stride
        djson[0]['parameters']['window_dilation_strides'] = dilation
        djson[0]['parameters']['padding_below_diff'] = padding
        djson[0]['op_type'] = 'QuantizeDepthwiseConv2dNative'
        djson[0]['parameters']['in_quantize_bit'] = 8
        djson[0]['parameters']['out_quantize_bit'] = 32
        djson[0]['parameters']["identifier_suffix"] = suffix
        djson[0]['parameters']["identifier_prefix"] ="Quantize"
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
        code = code + global_suffix
        djson[0]["code"] = code
        global_suffix += '\n'
        print(weight_shape)

        djson[0]['gridDim'][0]=int(out_size/512)
        djson[0]['blockDim'][0]=512
        file_name = 'efficient_' + conv.replace('.', '_')+'_'+suffix
        filepath = os.path.join(prefix, file_name)
        with open(filepath+'.json', 'w') as jf:
            json.dump(djson, jf) 
        with open(filepath+'.cu', 'w') as f:
            f.write(code)  

with open('./kernel/dot_add_relu.template', 'r') as cuf:
    dcu = cuf.read()

for conv in data:
    for suffix in ["", "BatchNormInference", "BatchNormInferenceSigmoid"]:

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
        djson[0]['parameters']['in_quantize_bit'] = 8
        djson[0]['parameters']['out_quantize_bit'] = 32
        djson[0]['parameters']["identifier_suffix"] = suffix
        djson[0]['parameters']["identifier_prefix"] ="Quantize"


        code = dcu
        M = in_shape[0]*in_shape[2]*in_shape[3] # N * H * W
        K = in_shape[1]
        N = out_shape[1]
        # if(M%16!=0):
        #     continue
        print(M,K,N)
        assert M%16 ==0
        if K % 16 > 0:
            K += 16 - K % 16
        assert K%16 == 0
        if N % 16 > 0:
            N += 16 - N % 16
        assert N%16 == 0
        code=code.replace('MGLOBAL_VALUE', str(M))
        code=code.replace('KGLOBAL_VALUE', str(K))
        code=code.replace('NGLOBAL_VALUE', str(N))
        code=code.replace('MTILES_VALUE', str(int(M/16)))
        code=code.replace('KTILES_VALUE', str(int(K/16)))
        code=code.replace('NTILES_VALUE', str(int(N/16)))
        code=code + global_suffix
        djson[0]["code"] = code
        global_suffix += '\n'
        print(weight_shape)


        djson[0]['gridDim'][0]=68
        djson[0]['blockDim'][0] = 256
        file_name = 'conv1x1_'+str(M)+'_'+str(K)+'_'+str(N)+'_'+suffix
        filepath = os.path.join(prefix, file_name)
        with open(filepath+'.json', 'w') as jf:
            json.dump(djson, jf) 
        with open(filepath+'.cu', 'w') as f:
            f.write(code)  