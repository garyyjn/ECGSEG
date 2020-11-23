import torch
from torch import Tensor
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional

class seqECGNET(nn.Module):
    def __init__( self,
                  CNNModule = 'ResNet50', CNNConfigs = [],
                  SEQModule = 'LSTM',  RNNConfigs = [],
                  feature_number = 50, num_class = 2):
        if CNNModule == 'ResNet50':


