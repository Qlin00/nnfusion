# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
This script helps you to insert specific kernels (specs in json files) into our kernel_db
Some more explains to customize it for your needs
    param_list : specify the parameters and their data types for a certain type of kernel
    gen_key    : generate identifier of kernel specification, including I/O shapes, types and more
    gen_config : convert kernel specification to the db format 
    insert_db  : insert the parsed kernel into kernel db
"""

import json
import sys
import sqlite3
import os
import math

from cuparse import parse as cu_code_parse
from ccparse import parse as cc_code_parse
from profile import prepare_file, log_sync, profile, prod

db_path = os.environ['HOME'] + "/.cache/nnfusion/"
db_name = "kernel_cache.db"


# Todo: re-org operator definition to oop and coordinate to NNFusion
param_list = {
    "Convolution": {
        'symbol': ['input0', 'input1', 'output0'],
        'dtype': ['float*', 'float*', 'float*']
    },
    "MaxPool": {
        'symbol': ['input0', 'output0'],
        'dtype': ['float*', 'float*']
    },
    "Relu": {
        'symbol': ['input0', 'output0'],
        'dtype': ['float*', 'float*']
    },
    "Dot": {
        'symbol': ['input0', 'input1', 'output0'],
        'dtype': ['float*', 'float*', 'float*']
    },
    "BitConverter":{
        'symbol': ['input0', 'input1', 'input2', 'output0'],
        'dtype': ['float*', 'float*', 'float*', "float*"]
    },
    "QuantizeDot": {
        'symbol': ['input0', 'input1', 'input2', 'input3', 'input4', 'input5', 'input6', 'output0'],
        'dtype': ['float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', "float*"]
    },
    "QuantizeDotAdd": {
        'symbol': ['input0', 'input1', 'input2', 'input3', 'input4', 'input5', 'input6', 'input7', 'output0'],
        'dtype': ['float*', 'float*', 'float*', 'float*', 'float*', 'float*', "float*", 'float*', "float*"]
    },
    "BlockQuantizeDotAdd": {
        'symbol': ['input0', 'input1', 'input2', 'input3', 'input4', 'input5', 'input6', 'output0'],
        'dtype': ['float*', 'float*', 'float*', 'float*', 'float*', 'float*', "float*", "float*"]
    },
    "Fused_Convolution_Relu": {
        'symbol': ['input0', 'input1', 'output0'],
        'dtype': ['float*', 'float*', 'float*']
    },
    "Fused_Convolution_Batchnorm": {
        'symbol': ['input0', 'input1', 'output0', 'input2'],
        'dtype': ['float*', 'float*', 'float*', 'float*']
    },
    "Fused_Convolution_Batchnorm_Relu": {
        'symbol': ['input0', 'input1', 'output0', 'input2'],
        'dtype': ['float*', 'float*', 'float*', 'float*']
    },
    "Fused_Convolution_Add_Relu": {
        'symbol': ['input0', 'input1', 'output0', 'input2'],
        'dtype': ['float*', 'float*', 'float*', 'float*']
    },
    "AvgPool": {
        'symbol': ['input0', 'output0'],
        'dtype': ['float*', 'float*']
    },
    "DepthwiseConv2dNative": {
        'symbol': ['input0', 'input1', 'output0'],
        'dtype': ['float*', 'float*', 'float*']
    },
    "QuantizeDepthwiseConv2dNative": {
        'symbol': ['input%d'%i for i in range(5)] +['output0'],
        'dtype': ['float*']*6
    },
    "QuantizeConvolution": {
        'symbol': ['input0', 'input1', 'input2', 'input3', 'input4', 'input5', 'input6', 'input7', 'output0'],
        'dtype': ['float*', 'float*', 'float*', 'float*', 'float*', 'float*', "float*", 'float*', "float*"]
    },
    "BlockQuantizeConvolution": {
        'symbol': ['input0', 'input1', 'input2', 'input3', 'input4', 'input5', 'input6', 'output0'],
        'dtype': ['float*', 'float*', 'float*', 'float*', 'float*', 'float*', "float*", "float*"]
    },
    "SparseDot": {
        # activation, w_val, w_row, w_col, c
        'symbol': ['input0', 'input1', 'input2', 'input3', 'input4','output0'],
        'dtype': ['float*', 'float*', 'float*', 'float*', 'float*', 'float*']
    }
}

conv_augmented = ["Fused_Convolution_Batchnorm",
                  "Fused_Convolution_Batchnorm_Relu", "Fused_Convolution_Add_Relu"]
conv_family = ["Convolution", "Fused_Convolution_Relu"] + conv_augmented
depthwise_conv_augmented = ["Fused_DepthwiseConv2dNative_Batchnorm",
                            "Fused_DepthwiseConv2dNative_Batchnorm_Relu", "Fused_DepthwiseConv2dNative_Add_Relu"]
depthwise_conv_family = ["DepthwiseConv2dNative", 
                         "Fused_DepthwiseConv2dNative_Relu"] + conv_augmented

def gen_config(op_type, kernel, shared_memory, num_sync):
    # the entries to retrive parameters depend on spec of json files
    config = {
        "op_type": op_type,
        "function_body": "",
        "shared_memory": shared_memory,
        "num_sync": num_sync,
        "blockDim": kernel["blockDim"],
        "gridDim": kernel["gridDim"],
    }
    if "dynamic_shared_memory" in kernel:
        config["dynamic_shared_memory"] = kernel["dynamic_shared_memory"]
    if "kernel_identifier" in kernel:
        config["kernel_identifier"] = kernel["kernel_identifier"]
    if op_type in conv_family + depthwise_conv_family:
        config["in_shape"] = [kernel["parameters"]
                              ["input_shape"], kernel["parameters"]["filter_shape"]]
        config["out_shape"] = [kernel["parameters"]["output_shape"]]
        config["parameters"] = {
            "window_movement_strides": kernel["parameters"]["window_movement_strides"],
            "window_dilation_strides": kernel["parameters"]["window_dilation_strides"],
            "padding_below_diff": kernel["parameters"]["padding_below_diff"]
        }
        if op_type in conv_augmented:
            config["in_shape"].append(config["out_shape"][0])
            config[
                "function_signature"] = "extern \"C\" __global__  void (float* input0, float* input1, float* input2, float* output0)"
        else:
            config["function_signature"] = "extern \"C\" __global__  void (float* input0, float* input1, float* output0)"
    elif (op_type == "Dot"):
        config["in_shape"] = [kernel["parameters"]
                              ["arg0_shape"], kernel["parameters"]["arg1_shape"]]
        config["out_shape"] = [kernel["parameters"]["out_shape"]]
        config[
            "function_signature"] = "extern \"C\" __global__  void (float* __restrict__ input0,  float* __restrict__ input1,  float* __restrict__ output0)"
    elif ("QuantizeDot" in op_type):
        config["in_shape"] = []
        for i in range(100):
            input_key = "arg%d_shape" % i 
            if input_key in kernel["parameters"]:
                config["in_shape"].append(kernel["parameters"][input_key])
        
        config["out_shape"] = [kernel["parameters"]["output_shape"]]
        # import pdb; pdb.set_trace()
        # in_paranames = ','.join(['float* __restrict__ input%d'%i for i in range(len(config["in_shape"]))])
        input_paras = ['float* __restrict__ input%d'%i for i in range(len(param_list[op_type]['symbol'])-1)]
        config[
            "function_signature"] = "extern \"C\" __global__  void (%s, float* __restrict__ output0)" % ','.join(input_paras)
    elif (op_type == 'SparseDot'):
        config["in_shape"] = []
        for i in range(100):
            input_key = f"arg{i}_shape"
            if input_key in kernel["parameters"]:
                config["in_shape"].append(kernel["parameters"][input_key])

        config["out_shape"] = [kernel["parameters"]["out_shape"]]
        in_paranames = ','.join(['float* __restrict__ input%d'%i for i in range(len(config["in_shape"]))])
        config[
            "function_signature"] = "extern \"C\" __global__  void (%s, float* __restrict__ output0)" % in_paranames
    elif (op_type == "Relu"):
        config["in_shape"] = [kernel["parameters"]["input_shape"]]
        config["out_shape"] = [kernel["parameters"]["output_shape"]]
        config["function_signature"] = "extern \"C\" __global__  void (float* input0, float* output0)"
    elif (op_type == "AvgPool" or op_type == "MaxPool"):
        config["in_shape"] = [kernel["parameters"]["input_shape"]]
        config["out_shape"] = [kernel["parameters"]["output_shape"]]
        config["function_signature"] = "extern \"C\" __global__  void (float* input0, float* output0)"
        config["parameters"] = {
            "window_shape": kernel["parameters"]["window_shape"],
            "window_stride": kernel["parameters"]["window_stride"],
            "padding_below": kernel["parameters"]["padding_below"]
        }
    elif (op_type == 'BitConverter'):
        config["in_shape"] = []
        for i in range(100):
            input_key = "arg%d_shape" % i 
            if input_key in kernel["parameters"]:
                config["in_shape"].append(kernel["parameters"][input_key])
        config["out_shape"] = [kernel["parameters"]["out_shape"]]
        in_paranames = ','.join(['float* __restrict__ input%d'%i for i in range(len(config["in_shape"]))])
        config[
            "function_signature"] = "extern \"C\" __global__  void (%s, float* __restrict__ output0)" % in_paranames
    elif (op_type=="QuantizeDepthwiseConv2dNative"):
        if "identifier_suffix" in kernel["parameters"]:
            config["identifier_suffix"] = kernel["parameters"]["identifier_suffix"]
        if 'identifier_prefix' in kernel['parameters']:
            config['identifier_prefix'] = kernel['parameters']['identifier_prefix']
        config["in_quantize_bit"] = kernel["parameters"]["in_quantize_bit"]
        config["out_quantize_bit"] = kernel["parameters"]["out_quantize_bit"]
        config["in_shape"] = [kernel["parameters"]
                              ["input_shape"], kernel["parameters"]["filter_shape"]]
        config["out_shape"] = [kernel["parameters"]["output_shape"]]
        config["parameters"] = {
            "window_movement_strides": kernel["parameters"]["window_movement_strides"],
            "window_dilation_strides": kernel["parameters"]["window_dilation_strides"],
            "padding_below_diff": kernel["parameters"]["padding_below_diff"]
        }
    
        # config["in_shape"].append(config["out_shape"][0])
        input_paras = ['float* __restrict__ input%d'%i for i in range(5)]
        out_paras = ['float* __restrict__ output0']
        config[
            "function_signature"] = "extern \"C\" __global__  void (%s)" %(','.join(input_paras+out_paras))


    elif ("QuantizeConvolution" in op_type):
        if "identifier_suffix" in kernel["parameters"]:
            config["identifier_suffix"] = kernel["parameters"]["identifier_suffix"]
        if 'identifier_prefix' in kernel['parameters']:
            config['identifier_prefix'] = kernel['parameters']['identifier_prefix']
        config["in_quantize_bit"] = kernel["parameters"]["in_quantize_bit"]
        config["out_quantize_bit"] = kernel["parameters"]["out_quantize_bit"]
        config["in_shape"] = [kernel["parameters"]
                              ["input_shape"], kernel["parameters"]["filter_shape"]]
        config["out_shape"] = [kernel["parameters"]["output_shape"]]
        config["parameters"] = {
            "window_movement_strides": kernel["parameters"]["window_movement_strides"],
            "window_dilation_strides": kernel["parameters"]["window_dilation_strides"],
            "padding_below_diff": kernel["parameters"]["padding_below_diff"]
        }
        # config["in_shape"].append(config["out_shape"][0])
        input_paras = ['float* __restrict__ input%d'%i for i in range(len(param_list[op_type]['symbol'])-1)]
        out_paras = ['float* __restrict__ output0']
        config[
            "function_signature"] = "extern \"C\" __global__  void (%s)" %(','.join(input_paras+out_paras))


    else:
        raise ("not implemented")

    return config


def insert_db(name, resource, platform="CUDA_GPU", tags="", profile="Tesla V100-PCIE-16GB:1"):
    # Todo: More tags could be used to store multiple implementations with the same kernel specs
    in_file = open(name + ".cu")
    json_file = open(name + ".json")

    data = json.load(json_file)
    block_function_body = in_file.read()
    data["block_function_body"] = block_function_body

    key = data["function_body"]

    op_type = data["op_type"]
    source = "External"
    device_type = platform

    attributes_dict = {}
    attributes_dict.update({"input_shape": data["in_shape"]})
    attributes_dict.update({"output_shape": data["out_shape"]})
    if data.get("parameters") != None:
        attributes_dict.update({"parameters": data["parameters"]})
    attributes = json.dumps(attributes_dict)

    function_dict = {}
    function_dict.update({"function_signature": data["function_signature"]})
    function_dict.update({"function_body": data["function_body"]})
    function_dict.update({"grid_dim": data["gridDim"]})
    function_dict.update({"block_dim": data["blockDim"]})
    if "dynamic_shared_memory" in data:
        function_dict.update({"dynamic_shared_memory": data["dynamic_shared_memory"]})
    function_dict.update({"block_function_body": data["block_function_body"]})
    function_dict.update({"shared_memory": data["shared_memory"]})
    function_dict.update({"num_syncthreads": data["num_syncthreads"]})
    function = json.dumps(function_dict)

    miscs_dict = {}
    profile_dict = {"time": profile, "resource": resource}
    miscs_dict.update({"external_profile": profile_dict})
    miscs = json.dumps(miscs_dict)

    conn = sqlite3.connect(db_path + db_name)
    c = conn.cursor()

    create_sql = "create table if not exists KernelCache (\
            Key        TEXT NOT NULL,\
            Identifier TEXT NOT NULL,\
            OpType     TEXT NOT NULL,\
            Attributes TEXT DEFAULT '',\
            Source     TEXT DEFAULT 'External',\
            DeviceType TEXT NOT NULL,\
            Function   TEXT NOT NULL,\
            Tags       TEXT DEFAULT '',\
            Miscs      TEXT DEFAULT '',\
            PRIMARY KEY(Key)\
            )"

    c.execute(create_sql)

    # identifier = gen_key(data)
    identifier = data['kernel_identifier']
    print(identifier)
    # overwrite the same implementation
    c.execute("DELETE FROM KernelCache WHERE Key = ?", (key,))
    c.execute("INSERT INTO KernelCache (Key,Identifier,OpType,Attributes,Source,DeviceType,Function,Tags,Miscs) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
              (key, identifier, op_type, attributes, source, device_type, function, tags, miscs))
    conn.commit()
    conn.close()


if __name__ == '__main__':
    if not os.path.isdir(db_path):
        os.mkdir(db_path)
    json_file = open(sys.argv[1])
    kernels = json.load(json_file)

    if len(sys.argv)>2:
        platform = sys.argv[2]
    else:
        platform = "CUDA_GPU"
    # input json file could contain one or more kernels
    if "op_type" in kernels:
        kernels = [kernels]

    code_parse = cu_code_parse
    if "CPU" in platform:
        code_parse = cc_code_parse
        print('Using cpu code parse')
        # import pdb; pdb.set_trace()

    for kernel in kernels:
        op_type = kernel["op_type"]

        # parse and clean up the cuda
        # code to get some specific information
        # import ipdb; ipdb.set_trace()
        
        func_body, shared_memory, new_code, sync_code, signature = code_parse(
            kernel["code"], param_list[op_type])

        config = gen_config(op_type, kernel, shared_memory, num_sync=0)

        prepare_file(signature, sync_code, config,
                     db_path + "profile/", parse=True)
        
        num_sync = 0
        config["num_syncthreads"] = num_sync
        config["function_body"] = func_body

        # feel free to customize the repo name you want
        name = kernel["tvm_func_name"].replace("_kernel0", "")
        operator_path = db_path + op_type + "_db/"
        if not os.path.isdir(operator_path):
            os.mkdir(operator_path)
        with open(operator_path + name + ".json", "w+") as f:
            json.dump(config, f)
        with open(operator_path + name + ".cu", "w+") as f:
            f.write(new_code)

        default_tags = ""
        default_tags += "KernelEmitter,CudaEmitter"
        # if (op_type == "Dot" or  "QuantizeDot" in op_type):
        #     # Todo: move the transpose information into identifier
        #     default_tags += kernel["parameters"]["transpose_A"] * \
        #         ",transA" + kernel["parameters"]["transpose_B"]*",transB"

        # apply rules that every 32 threads will be formed as a warp
        resource = math.ceil(
            prod(config["blockDim"])/32)*32 * prod(config["gridDim"])

        prepare_file(signature, kernel["code"], config, db_path + "profile/")
        # profile_info = profile(signature, db_path + "profile/")
        profile_info = "Skip the profile"
        print(profile_info, resource, config["num_syncthreads"])
        insert_db(operator_path + name, resource, platform=platform,
                  tags=default_tags, profile=profile_info)
