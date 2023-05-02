import pandas as pd
import json
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn

class CONFIG:
    ROOT_DIR = '/kaggle/input/asl-signs'
    TRAIN_DATA = '/kaggle/input/asl-signs/train.csv'
    EXTENDED_TRAIN_DATA = '/kaggle/input/gislr-extended-train-dataframe/extended_train.csv'
    TRAIN_DIR = '/kaggle/input/asl-signs/train_landmark_files'
    PREDICTION_INDEX_MAP = '/kaggle/input/asl-signs/sign_to_prediction_index_map.json'
    
    LANDMARKS_PER_FRAME = 543
    AVERAGE_FRAME = 37
    SEED = 11
    
    CONTOURS = [0, 9, 11, 13, 14, 17, 117, 118, 119, 199, 346, 347, 348]

    
    LANDMARKS_IDX = {"contours" : list(set(CONTOURS)),
                     "left_hand" : np.arange(468, 489).tolist(),
                     "upper_body" : np.arange(489, 511).tolist(),
                     "right_hand" : np.arange(522, 543).tolist()}
    
def read_json_file(json_path):
    with open(json_path, "rb") as f:
        json_data = json.load(f)
    
    return json_data

def load_landmarks(pq_path):
    data_columns = ['x', 'y', 'z']
    
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = int(len(data) / CONFIG.LANDMARKS_PER_FRAME)
    data = data.values.reshape(n_frames, CONFIG.LANDMARKS_PER_FRAME, len(data_columns))
    
    return data.astype(np.float32)

train_df = pd.read_csv(CONFIG.EXTENDED_TRAIN_DATA)

label_map = read_json_file(CONFIG.PREDICTION_INDEX_MAP)

train_df['label'] = train_df['sign'].map(label_map)

class FeaturePreprocess(nn.Module):
    def __init__(self):
        super(FeaturePreprocess, self).__init__()
        
    def forward(self, x):
        n_frames = x.shape[0]

        # Normalization to a common mean
        x = x - x[~torch.isnan(x)].mean(0,keepdim=True) 
        x = x / x[~torch.isnan(x)].std(0, keepdim=True)

        # Landmarks reduction
        contours = x[:, CONFIG.LANDMARKS_IDX['contours']]
        left_hand = x[:, CONFIG.LANDMARKS_IDX['left_hand']]
        pose = x[:, CONFIG.LANDMARKS_IDX['upper_body']]
        right_hand = x[:, CONFIG.LANDMARKS_IDX['right_hand']]
       
        x = torch.cat([contours, left_hand, pose, right_hand], 1) # (n_frames, 192, 3)
        
        # Replace nan with 0
        x[torch.isnan(x)] = 0
        
        x = x.permute(2, 1, 0) #(3, 77, n_frames)
        if n_frames < CONFIG.AVERAGE_FRAME:
            x = F.interpolate(x, size=(CONFIG.AVERAGE_FRAME), mode= 'linear')
        else:
            x = F.interpolate(x, size=(CONFIG.AVERAGE_FRAME), mode= 'nearest-exact')
        
        return x.permute(2, 1, 0)

feature_preprocess = FeaturePreprocess()

def convert_row(row):
    x = torch.tensor(load_landmarks(row[1].path))
    x = feature_preprocess(x).cpu().numpy()
    
    return x, row[1].label

def convert_and_save_data(df, data_path, label_path, save_data=False):
    total = df.shape[0]
    npdata = np.zeros((total, 37, 77 ,3))
    nplabels = np.zeros(total)
    for i, row in tqdm(enumerate(df.iterrows()), total=total):
        x, y = convert_row(row)
        npdata[i,:,:,:] = x
        nplabels[i] = y
    
    if save_data:
        np.save(data_path, npdata)
        np.save(label_path, nplabels)
    
    return npdata, nplabels

features_path = "/kaggle/working/feature_data.npy"
labels_path = "/kaggle/working/feature_labels.npy"

features, labels = convert_and_save_data(train_df, features_path, labels_path)
