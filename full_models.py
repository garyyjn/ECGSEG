from video_feature_extraction_models import *
import sequencial_models
from torch import nn
class full_model(nn.Module):
    def __init__(self, conv_features):
        self.feature_selector = T3d(conv_features)
        self.sequencial_inf = nn.LSTM(conv_features)

    def forward(self, x):
        h = self.feature_selector(x)
        y = self.sequencial_inf(h)

