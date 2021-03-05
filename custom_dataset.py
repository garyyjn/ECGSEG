import numpy as np
import csv
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import skvideo.io
import os

class kinetics_dataset(Dataset):
    def __init__(self, annotations, video_dir, transform=None):
        self.action_annotations = pd.read_csv(annotations, index_col = 0)
        self.video_dir = video_dir
        self.transform = transform

    def __len__(self):
        return len(self.action_annotations)

    def __getitem__(self, idx):#todo preprocess videos, currently too slow
        if torch.is_tensor(idx):
            idx = idx.tolist()
        item_row = self.action_annotations.iloc[idx]
        video_name = item_row['video_name']
        video_matrix = self.video2matrix(path = os.path.join(self.video_dir, video_name))
        video_matrix = video_matrix[0:150,:,:,:] #here we reshape this video matrix into 5 segments of 30 frames * h * w * colors
        video_matrix = video_matrix.reshape((5,30,240,320,3))
        #if skip_pattern:
           # video_matrix = video_matrix[:,skip_pattern,:,:,:]
        action = item_row['action']
        sample = {'video': video_matrix, 'action':action}
        #sample = {'video_name': video_matrix, 'action':action}
        if self.transform:
            sample = self.transform(sample)
        return sample


    def video2matrix(self, path):  # returns shape: frames x height x width x color_dims
        videodata = skvideo.io.vread(path)
        return np.array(videodata)

class echonet_dataset(Dataset):#still have to take care of volumn annotations and too short videos
    def __init__(self, annotations, video_dir, transform=None):
        self.volumn_annotations = None#pd.read_csv(annotations, index_col = 0)
        self.video_dir = video_dir
        self.transform = transform

    def __len__(self):
        return len([name for name in self.video_dir])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        item_row = self.action_annotations.iloc[idx]
        video_name = item_row['video_name']
        video_matrix = self.video2matrix(path = os.path.join(self.video_dir, video_name))
        video_matrix = video_matrix[0:150,:,:,:] #here we reshape this video matrix into 5 segments of 30 frames * h * w * colors
        video_matrix = video_matrix.reshape((5,30,240,320,3))
        action = item_row['action']
        sample = {'video': video_matrix, 'action':action}
        #sample = {'video_name': video_matrix, 'action':action}
        if self.transform:
            sample = self.transform(sample)
        return sample


    def video2matrix(self, path):  # returns shape: frames x height x width x color_dims
        videodata = skvideo.io.vread(path)
        return np.array(videodata)

'''
kd = kinetics_dataset('video_annotations', 'UCF-101-Flattened')
print(kd.__len__())
item1 = kd.__getitem__(1)
print(item1['video'].shape)
'''