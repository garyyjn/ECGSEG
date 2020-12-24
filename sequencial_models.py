import torch
import torch.nn
from torch import nn
class simple_linear(nn.Module):
    def __init__(self, segment_numbers, segment_features, class_number):
        super(simple_linear, self).__init__()
        self.fc1 = nn.Linear(segment_numbers*segment_features, 200)
        self.fc2 = nn.Linear(200,class_number)
        self.soft_max = nn.Softmax()
        self.segment_number = segment_numbers
        self.segment_features = segment_features
    def forward(self, x):
        #x has dim: batch * segments * features
        #output batch*101 classes
        x = x.reshape([-1, self.segment_number*self.segment_features])
        h = self.fc1(x)
        h = self.fc2(h)
        y = self.soft_max(h)
        return y
