import torch
import torch.nn
from torch import nn
class T3d(nn.Module):
    '''
    3d3 model from paper - Learning Spatiotemporal Features with 3D Convolutional Networks
    reimplemented and slightly changed to produce a vector of features for each time frame
    '''
    def __init__(self, feature_num):
        '''
        inchannel = 3 for colors?
        outchannel = feature_num
        '''
        super(T3d, self).__init__()
        self.feature_num = feature_num
        self.Conv1a = torch.nn.Conv3d(3, 64, kernel_size = (3,3,3), stride=1, padding=(1, 1, 1))
        self.Pool1 = torch.nn.MaxPool3d(kernel_size = (1,2,2), stride = (1,2,2))
        self.Conv2a = torch.nn.Conv3d(64, 128, kernel_size=(3,3,3), stride = 1, padding=(1, 1, 1))
        self.Pool2 = torch.nn.MaxPool3d(kernel_size=(2,2,2), stride = (2,2,2))
        self.Conv3a = torch.nn.Conv3d(128, 256, kernel_size=(3,3,3), stride=1, padding=(1, 1, 1))
        self.Conv3b = torch.nn.Conv3d(256, 256, kernel_size=(3,3,3), stride=1, padding=(1, 1, 1))
        self.Pool3 = torch.nn.MaxPool3d(kernel_size=(2,2,2), stride = (2,2,2))
        self.Conv4a = torch.nn.Conv3d(256, 512, kernel_size=(3,3,3), stride=1, padding=(1, 1, 1))
        self.Conv4b = torch.nn.Conv3d(512, 512, kernel_size=(3,3,3), stride=1, padding=(1, 1, 1))
        self.Pool4 = torch.nn.MaxPool3d(kernel_size=(2,2,2), stride = (2,2,2))
        self.Conv5a = torch.nn.Conv3d(512, 512, kernel_size=(3,3,3), stride=1, padding=(1, 1, 1))
        self.Conv5b = torch.nn.Conv3d(512, 512, kernel_size=(3,3,3), stride=1, padding=(1, 1, 1))
        self.Pool5 = torch.nn.MaxPool3d(kernel_size=(2,2,2), stride = (2,2,2), padding=(0, 1, 0))
        self.Pool5b = torch.nn.MaxPool3d(kernel_size=(1,2,2), stride = (1,2,2), padding=(0, 0, 0))
        self.fc6 = torch.nn.Linear(10240, 4096)
        self.fc7 = torch.nn.Linear(4096, 4096)
        self.fc8 = torch.nn.Linear(4096, feature_num)
        self.sm = torch.nn.Softmax()

    def forward(self, x):
        h = self.Conv1a(x)
        h = self.Pool1(h)
        print("After p1: " + str(h.shape))
        h = self.Conv2a(h)
        h = self.Pool2(h)
        print("After p2: " + str(h.shape))
        h = self.Conv3a(h)
        h = self.Conv3b(h)
        h = self.Pool3(h)
        print("After p3: " + str(h.shape))
        h = self.Conv4a(h)
        h = self.Conv4b(h)
        h = self.Pool4(h)
        print("After p4: " + str(h.shape))
        h = self.Conv5a(h)
        h = self.Conv5b(h)
        h = self.Pool5(h)
        print("After p5: " + str(h.shape))
        h = self.Pool5b(h)
        print("After p5b: " + str(h.shape))
        pre_fc_shape = h.shape
        batch_frame_size = pre_fc_shape[0]
        h = h.reshape([batch_frame_size, -1])
        h = self.fc6(h)
        h = self.fc7(h)
        h = self.fc8(h)
        features = self.sm(h)
        return features

class T3d_weak(nn.Module):
    '''
    3d3 model from paper - Learning Spatiotemporal Features with 3D Convolutional Networks
    reimplemented and slightly changed to produce a vector of features for each time frame
    '''
    def __init__(self, feature_num):
        '''
        inchannel = 3 for colors?
        outchannel = feature_num
        '''
        super(T3d_weak, self).__init__()
        self.feature_num = feature_num
        self.Conv1a = torch.nn.Conv3d(3, 32, kernel_size = (3,3,3), stride=1, padding=(1, 1, 1))
        self.Pool1 = torch.nn.MaxPool3d(kernel_size = (1,2,2), stride = (1,2,2))
        self.Conv2a = torch.nn.Conv3d(32, 64, kernel_size=(3,3,3), stride = 1, padding=(1, 1, 1))
        self.Pool2 = torch.nn.MaxPool3d(kernel_size=(2,2,2), stride = (2,2,2))
        self.Conv3a = torch.nn.Conv3d(64, 128, kernel_size=(3,3,3), stride=1, padding=(1, 1, 1))
        self.Conv3b = torch.nn.Conv3d(128, 128, kernel_size=(3,3,3), stride=1, padding=(1, 1, 1))
        self.Pool3 = torch.nn.MaxPool3d(kernel_size=(2,2,2), stride = (2,2,2))
        self.Conv4a = torch.nn.Conv3d(128, 256, kernel_size=(3,3,3), stride=1, padding=(1, 1, 1))
        self.Conv4b = torch.nn.Conv3d(256, 256, kernel_size=(3,3,3), stride=1, padding=(1, 1, 1))
        self.Pool4 = torch.nn.MaxPool3d(kernel_size=(2,2,2), stride = (2,2,2))
        self.Conv5a = torch.nn.Conv3d(256, 256, kernel_size=(3,3,3), stride=1, padding=(1, 1, 1))
        self.Conv5b = torch.nn.Conv3d(256, 256, kernel_size=(3,3,3), stride=1, padding=(1, 1, 1))
        self.Pool5 = torch.nn.MaxPool3d(kernel_size=(2,2,2), stride = (2,2,2), padding=(0, 1, 0))
        self.Pool5b = torch.nn.MaxPool3d(kernel_size=(1,2,2), stride = (1,2,2), padding=(0, 0, 0))
        self.fc6 = torch.nn.Linear(5120, 2000)
        self.fc7 = torch.nn.Linear(2000, 2000)
        self.fc8 = torch.nn.Linear(2000, feature_num)
        self.sm = torch.nn.Softmax()

    def forward(self, x):
        h = self.Conv1a(x)
        h = self.Pool1(h)
        #print("After p1: " + str(h.shape))
        h = self.Conv2a(h)
        h = self.Pool2(h)
        #print("After p2: " + str(h.shape))
        h = self.Conv3a(h)
        h = self.Conv3b(h)
        h = self.Pool3(h)
        #print("After p3: " + str(h.shape))
        h = self.Conv4a(h)
        h = self.Conv4b(h)
        h = self.Pool4(h)
        #print("After p4: " + str(h.shape))
        h = self.Conv5a(h)
        h = self.Conv5b(h)
        h = self.Pool5(h)
        #print("After p5: " + str(h.shape))
        h = self.Pool5b(h)
        #print("After p5b: " + str(h.shape))
        pre_fc_shape = h.shape
        batch_frame_size = pre_fc_shape[0]
        h = h.reshape([batch_frame_size, -1])
        h = self.fc6(h)
        h = self.fc7(h)
        h = self.fc8(h)
        features = self.sm(h)
        return features

class T3d_weak_skipped(nn.Module):
    '''
    3d3 model from paper - Learning Spatiotemporal Features with 3D Convolutional Networks
    reimplemented and slightly changed to produce a vector of features for each time frame
    '''

    def __init__(self, feature_num, skip_pattern):
        '''
        inchannel = 3 for colors?
        outchannel = feature_num
        '''
        super(T3d_weak_skipped, self).__init__()
        self.feature_num = feature_num
        self.skip_pattern = skip_pattern
        self.Conv1a = torch.nn.Conv3d(3, 32, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.Pool1 = torch.nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.Conv2a = torch.nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.Pool2 = torch.nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.Conv3a = torch.nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.Conv3b = torch.nn.Conv3d(128, 128, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.Pool3 = torch.nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.Conv4a = torch.nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.Conv4b = torch.nn.Conv3d(256, 256, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.Pool4 = torch.nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.Conv5a = torch.nn.Conv3d(256, 256, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.Conv5b = torch.nn.Conv3d(256, 256, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.Pool5 = torch.nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 0))
        self.Pool5b = torch.nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0))
        self.fc6 = torch.nn.Linear(5120, 2000)
        self.fc8 = torch.nn.Linear(2000, feature_num)
        self.sm = torch.nn.Softmax()

    def forward(self, x):
        h = self.Conv1a(x)
        h = self.Pool1(h)
        # print("After p1: " + str(h.shape))
        h = self.Conv2a(h)
        h = self.Pool2(h)
        # print("After p2: " + str(h.shape))
        h = self.Conv3a(h)
        h = self.Conv3b(h)
        h = self.Pool3(h)
        # print("After p3: " + str(h.shape))
        h = self.Conv4a(h)
        h = self.Conv4b(h)
        h = self.Pool4(h)
        # print("After p4: " + str(h.shape))
        h = self.Conv5a(h)
        h = self.Conv5b(h)
        h = self.Pool5(h)
        # print("After p5: " + str(h.shape))
        h = self.Pool5b(h)
        # print("After p5b: " + str(h.shape))
        pre_fc_shape = h.shape
        batch_frame_size = pre_fc_shape[0]
        h = h.reshape([batch_frame_size, -1])
        h = self.fc6(h)
        h = self.fc7(h)
        h = self.fc8(h)
        features = self.sm(h)
        return features