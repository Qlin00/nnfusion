from openvino.inference_engine import IECore
import time
import numpy as np
 
def measure_time(model, dummy_input, runtimes=200):
    times = []
    for runtime in range(runtimes):
        start = time.time()
        re = model.infer(dummy_input)
        end = time.time()
        times.append(end-start)
    _drop = int(runtimes * 0.1)
    mean = np.mean(times[_drop:-1*_drop])
    std = np.std(times[_drop:-1*_drop])
    return mean*1000, std*1000

ie = IECore()
net = ie.read_network('bert_ori.onnx')
exec_net_onnx = ie.load_network(network=net, device_name="CPU")

dummy_input = {}
for name in net.input_info:
    if net.input_info[name].tensor_desc.precision == 'I64':
        dummy_input[name] = np.random.randint(0,128,size=net.input_info[name].tensor_desc.dims,dtype=np.int64)

print(measure_time(exec_net_onnx, dummy_input))