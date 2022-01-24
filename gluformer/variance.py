import torch
import torch.nn as nn
import torch.nn.functional as F

class Variance(nn.Module):
  def __init__(self, d_model, r_drop, len_seq):
    super(Variance, self).__init__()

    self.proj1 = nn.Linear(d_model, 1)
    self.dropout = nn.Dropout(r_drop) 
    self.activ1 = nn.ReLU()
    # + 1 (for seq) for embedded person token
    self.proj2 = nn.Linear(len_seq+1, 1)
    self.activ2 = nn.Tanh()

  def forward(self, x):
    x = self.proj1(x)
    x = self.activ1(x)
    x = self.dropout(x)
    x = x.transpose(-1, 1)
    x = self.proj2(x)
    # scale to [-10, 10] range
    x = 10 * self.activ2(x)
    return x