from video_feature_extraction_models import *
from sequencial_models import *
from torch import nn

class full_model_simple(nn.Module):
    def __init__(self, segment_number, conv_features, class_number):
        super(full_model_simple, self).__init__()
        self.feature_selector = T3d_weak(conv_features)#output batch*segment*feature_count
        self.sequencial_inf = simple_linear(segment_number, conv_features, class_number)
        self.segment_number = segment_number
        self.conv_features = conv_features
        self.class_number = class_number
    def forward(self, x):
        h = self.feature_selector(x)
        #h has dim batch*segment x segment_features - >  batch x segment x segment_features
        h = h.reshape([-1, self.segment_number, self.conv_features])
        y = self.sequencial_inf(h)
        return y