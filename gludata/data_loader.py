import os
import numpy as np
import pandas as pd
import pickle

import torch
from torch.utils.data import Dataset, DataLoader

class CGMData(Dataset):
  def __init__(self, path, flag='train', size=None):
    # size [seq_len, label_len, pred_len]
    # info
    self.seq_len = size[0]
    self.label_len = size[1]
    self.pred_len = size[2]
    
    # init
    self.path = path + flag + '_data.pkl'
    self.__read_data__()

  def __read_data__(self):
    with open(self.path, 'rb') as f:
      data = pickle.load(f)
    self.data = data

    len_segs = np.array([len(subj_seg[1]) for subj_seg in data])
    len_segs = len_segs - self.seq_len - self.pred_len + 1
    self.len_segs = np.insert(np.cumsum(len_segs), 0, 0)
  
  def __getitem__(self, index):

    idx_seg = np.argmax(self.len_segs > index) - 1
    seg = self.data[idx_seg]

    s_begin = index - self.len_segs[idx_seg]
    s_end = s_begin + self.seq_len
    r_begin = s_end - self.label_len
    r_end = r_begin + self.label_len + self.pred_len

    subj_id = seg[0]
    seq_x = seg[1][s_begin:s_end]
    seq_y = seg[1][r_begin:r_end]
    seq_x_mark = seg[2][s_begin:s_end]
    seq_y_mark = seg[2][r_begin:r_end]

    return subj_id, seq_x, seq_y, seq_x_mark, seq_y_mark
  
  def __len__(self):
    return self.len_segs[len(self.len_segs) - 1]
