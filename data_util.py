import numpy as np
import cv2
import skvideo.io
import pandas as pd
import os
from shutil import copy
import cv2
import matplotlib.pyplot as plt

def video2matrix(path):#returns shape: frames x height x width x color_dims
    videodata = skvideo.io.vread(path)
    return np.array(videodata)

def action2hot(action_list, aux_dict):
    target_matrix = np.zeros((len(action_list), 101))
    for i in range(len(action_list)):
        target_matrix[i, aux_dict[action_list[i]]] = 1
    return target_matrix

def action2label(action_list, aux_dict):
    target_list = []
    for i in range(len(action_list)):
        target_list.append( aux_dict[action_list[i]])
    return target_list

def getauxdict(original_UCF_path):
    aux_dict = {}
    for action_name in os.listdir(original_UCF_path):
        if(action_name not in aux_dict):
            aux_dict[action_name] = len(aux_dict)
    return aux_dict

def flattenFolder(original_UCF_path, dest_path, annotation_name, length_limit = 150): #flatten video folder and create annotation in csv
    aux_dic_list = []
    for folder_action_name in os.listdir(original_UCF_path):
        for video_name in os.listdir(os.path.join(original_UCF_path, folder_action_name)):
            full_video_path = os.path.join(original_UCF_path, folder_action_name, video_name)
            action_name = folder_action_name

            cap = cv2.VideoCapture(full_video_path)
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if(length < length_limit):
                continue
            else:
                copy(full_video_path, os.path.join(dest_path, video_name))
                aux_dic_list.append({'video_name':video_name, 'action':action_name})

    annotations = pd.DataFrame(aux_dic_list)
    annotations.to_csv(annotation_name)

def findminframe(path):
    min = 100000
    for video_name in os.listdir('UCF-101-Flattened'):
        cap = cv2.VideoCapture(os.path.join(path, video_name))
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if(min > length):
            min = length
            print("current min: " + str(min))
    print("min video frames: " + str(min))

def analyzeframe(path):
    frame_list = []
    for video_name in os.listdir('UCF-101-Flattened'):
        cap = cv2.VideoCapture(os.path.join(path, video_name))
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_list.append(length)
    bar_data = np.zeros((20), dtype=int)
    for frame_c in frame_list:
        if frame_c > 190:
            bar_data[19]+=1
        else:
            bar_data[int(frame_c/10)] += 1
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    frame_cata = [str(i)*10 for i in range(20)]
    ax.bar(frame_cata, bar_data)
    plt.show()
#analyzeframe('UCF-101-Flattened')
#findminframe('UCF-101-Flattened')
#flattenFolder('UCF-101', 'UCF-101-Flattened-150', annotation_name='video_annotations_150',length_limit=150)
