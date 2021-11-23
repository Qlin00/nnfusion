from openvino.inference_engine import IECore
import time
import numpy as np
import onnxruntime as ort
import onnx 
from timeit import default_timer as timer
def measure_time(model, dummy_input, runtimes=200):
    times = []
    for runtime in range(runtimes):
        # start = time.time()
        start = timer()
        re = model.infer(dummy_input)
        end = timer()
        times.append(end-start)
    _drop = int(runtimes * 0.1)
    mean = np.mean(times[_drop:-1*_drop])
    std = np.std(times[_drop:-1*_drop])
    return mean*1000, std*1000

def measure_time_ort(ori_sess, dummy_input, runtimes=200):
    times = []
    for runtime in range(runtimes):
        start = time.time()
        ort_sess.run(None, dummy_input)
        end = time.time()
        times.append(end-start)
    _drop = int(runtimes * 0.1)
    mean = np.mean(times[_drop:-1*_drop])
    std = np.std(times[_drop:-1*_drop])
    return mean*1000, std*1000

ie = IECore()
net = ie.read_network('model.onnx')
exec_net_onnx = ie.load_network(network=net, device_name="CPU")

dummy_input = {}
for name in net.input_info:
    if net.input_info[name].tensor_desc.precision == 'I64':
        dummy_input[name] = np.random.randint(0,128,size=net.input_info[name].tensor_desc.dims,dtype=np.int64)
    elif net.input_info[name].tensor_desc.precision == 'FP32':
        dummy_input[name] = np.random.random(size=net.input_info[name].tensor_desc.dims).astype(np.float32)
        print(dummy_input[name].shape)
    else:
        import pdb; pdb.set_trace()
# import pdb; pdb.set_trace()

print(measure_time(exec_net_onnx, dummy_input))
# import pdb; pdb.set_trace()

#ort_sess = ort.InferenceSession('model.onnx')
#print(measure_time_ort(ort_sess, dummy_input))
