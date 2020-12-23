import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from custom_dataset import kinetics_dataset

kd = kinetics_dataset('video_annotations_150', 'UCF-101-Flattened-150')
print(kd.__len__())
dataloader = DataLoader(kd, batch_size=10, shuffle=True, num_workers=0)

for i_batch, sample_batched in enumerate(dataloader):
    print('batch#: ' + str(i_batch))
    print(sample_batched['action'])
    print(sample_batched['video'].shape)
