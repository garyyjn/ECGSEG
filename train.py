import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from custom_dataset import kinetics_dataset
from video_feature_extraction_models import T3d
from full_models import *
import torch.optim as optim
from data_util import getauxdict, action2hot, action2label
from time import sleep
from tqdm import tqdm

batch_size = 1
segment_number = 5
segment_feature = 50
class_number = 101

kd = kinetics_dataset('video_annotations_150', 'UCF-101-Flattened-150')
dataloader = DataLoader(kd, batch_size=batch_size, shuffle=True, num_workers=0)

model = full_model_simple(segment_number = segment_number, conv_features = 50, class_number = class_number)
if torch.cuda.is_available():
  dev = "cuda:0"
else:
  dev = "cpu"
print('running on' + str(dev))
model.cuda()
print('cuda check' + str(next(model.parameters()).is_cuda))

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.8)

auxdict = getauxdict('UCF-101')


def train_batch(model, optimizer, videos, labels):
    '''
    :param videos:n*segments, channels, frames, height, width
    :param labels:n*101 1/0
    '''
    optimizer.zero_grad()
    y = model(videos)
    loss = criterion(y, labels)
    loss.backward()
    optimizer.step()
    print("loss" + str(loss.item()))


for i_batch, sample_batched in tqdm(enumerate(dataloader)):
    print("batch# " + str(i_batch))
    video_matrix = sample_batched['video']
    n, segments, frames, height, width, channels = list(video_matrix.shape)
    train_shape = [n*segments, channels, frames, height, width]
    video_matrix = video_matrix.reshape(train_shape).float().cuda()
    action_list = sample_batched['action']
    labels = torch.tensor(action2label(list(action_list), auxdict), dtype=torch.long).cuda()
    train_batch(model, optimizer, video_matrix, labels)
    sleep(.1)


'''
some memory tuning stuf
def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

print(get_n_params(model))
'''