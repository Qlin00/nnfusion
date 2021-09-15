import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR, StepLR
import argparse
from nni.algorithms.compression.pytorch.pruning import LevelPruner


import time
import numpy as np
import copy



class MLPSmall(nn.Module):
    def __init__(self):
        super(MLPSmall, self).__init__()
        self.fc1 = nn.Linear(1024, 512)
        # linear layer (n_hidden -> hidden_2)
        self.fc2 = nn.Linear(512, 512)
        # linear layer (n_hidden -> 10)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 10)
        # dropout layer (p=0.2)
        # dropout prevents overfitting of data
        #self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # flatten image input
        x = x.view(-1, 1024)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        # add hidden layer, with relu activation function
        #x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
        #x = F.log_softmax(self.fc4(x), 1)
        return x


# dummy_input = torch.rand(1024, 1, 32, 32)

# model = MLP()
# torch.onnx.export(model, dummy_input, 'mlp_norelu.onnx', verbose=True)

# small_model = MLPSmall()
# torch.onnx.export(small_model, dummy_input, 'mlp_norelu_small.onnx', verbose=True)

def measure_time(model, data, runtimes=100):
    times = []
    sum=0
    with torch.no_grad():
        for runtime in range(runtimes):
            torch.cuda.synchronize()
            start = time.time_ns()
            out=model(*data)
            # sum+=out[0][0]
            torch.cuda.synchronize()
            end = time.time_ns()
            times.append(end-start)
    print(sum)
    _drop = int(runtimes * 0.1)
    mean = np.mean(times[_drop:-1*_drop])
    std = np.std(times[_drop:-1*_drop])
    return mean, std

model = MLP().cuda()
# model.load_state_dict(torch.load('./MLP_Finegrained/mlp_mid_finegrained_prune_0.980_acc_0.950.pth'))
dummy_input = torch.rand(1024, 1, 32, 32).cuda()
# import pdb; pdb.set_trace()
# model(dummy_input)
# torch.onnx.export(model, dummy_input, 'mlp_norelu_finegrained.onnx')
t_mean, t_std = measure_time(model, [dummy_input])
print(t_mean/1000.0)

