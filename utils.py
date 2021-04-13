import random
import time
import datetime
import sys
import numpy as np
import jittor as jt
from jittor import nn

class ReplayBuffer:
    def __init__(self, max_size=50):
        assert max_size > 0, "Empty buffer or trying to create a black hole. Be careful."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []

        for i in range(data.size(0)):
            element = data[i:i+1]
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return jt.contrib.concat(to_return, dim=0)

def cross_entropy_loss_no_reduction(output, target):
    if len(output.shape) == 4:
        c_dim = output.shape[1]
        output = output.transpose((0, 2, 3, 1))
        output = output.reshape((-1, c_dim))
    target = target.reshape((-1, ))
    target = target.broadcast(output, [1])
    target = target.index(1) == target
    
    output = output - output.max([1], keepdims=True)
    loss = output.exp().sum(1).log()
    loss = loss - (output*target).sum(1)
    return loss

class CrossEntropyLossNoReduction(nn.Module):
    def __init__(self):
        pass
    def execute(self, output, target):
        return cross_entropy_loss_no_reduction(output, target)

def normalize_tensor(in_feat, eps=1e-10):
    norm_factor = jt.sqrt(jt.sum(in_feat**2, 1, keepdims=True))
    return in_feat/(norm_factor+eps)