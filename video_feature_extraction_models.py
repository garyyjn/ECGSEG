import torch
import torch.nn
from torch import nn
class T3d(nn.Module):
    '''
    3d3 model from paper - Learning Spatiotemporal Features with 3D Convolutional Networks
    reimplemented to produce a vector of features for each time frame
    '''
    def __init__(self, feature_num):
        '''
        inchannel = 3 for colors?
        outchannel = feature_num
        '''
        self.feature_num = feature_num
        self.Conv1a = torch.nn.Conv3d(3, 64, kernel_size = (3,3,3), stride = 1)
        self.Pool1 = torch.nn.MaxPool3d(kernel_size = (1,2,2), stride = (2,2,2))
        self.Conv2a = torch.nn.Conv3d(64, 128, kernel_size=(3,3,3), stride = 1)
        self.Pool2 = torch.nn.MaxPool3d(kernel_size=(2,2,2))
        self.Conv3a = torch.nn.Conv3d(128, 256, kernel_size=(3,3,3), stride=1)
        self.Conv3b = torch.nn.Conv3d(256, 256, kernel_size=(3,3,3), stride=1)
        self.Pool3 = torch.nn.MaxPool3d(kernel_size=(2,2,2))
        self.Conv4a = torch.nn.Conv3d(256, 512, kernel_size=(3,3,3), stride=1)
        self.Conv4b = torch.nn.Conv3d(512, 512, kernel_size=(3,3,3), stride=1)
        self.Pool4 = torch.mnn.MaxPool3d(kernel_size=(2,2,2))
        self.Conv5a = torch.nn.Conv3d(512, 512, kernel_size=(3,3,3), stride=1)
        self.Conv5b = torch.nn.Conv3d(512, 512, kernel_size=(3,3,3), stride=1)
        self.Pool5 = torch.nn.MaxPool3d(kernel_size=(2,2,2))
        self.fc6 = torch.nn.Linear(8192, 4096)
        self.fc7 = torch.nn.Linear(4096, 4096)
        self.fc8 = torch.nn.Linear(4096, feature_num)

    def forward(self, x):
        h = self.Conv1a(x)
        h = self.Pool1(h)
        h = self.Conv2a(h)
        h = self.Pool2(h)
        h = self.Conv3a(h)
        h = self.Conv3b(h)
        h = self.pool3(h)
        h = self.Conv4a(h)
        h = self.Conv4b(h)
        h = self.pool4(h)
        h = self.Conv5a(h)
        h = self.Conv5b(h)
        h = self.Pool5(h)
        h = self.fc6(h)
        h = self.fc7(h)
        h = self.fc8(h)

        features = nn.Softmax(h)
        return features
