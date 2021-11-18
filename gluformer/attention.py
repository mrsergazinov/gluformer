import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class CausalConv1d(torch.nn.Conv1d):
  def __init__(self,
                in_channels,
                out_channels,
                kernel_size,
                stride=1,
                dilation=1,
                groups=1,
                bias=True):
    self.__padding = (kernel_size - 1) * dilation

    super(CausalConv1d, self).__init__(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=self.__padding,
        dilation=dilation,
        groups=groups,
        bias=bias)

  def forward(self, input):
    result = super(CausalConv1d, self).forward(input)
    if self.__padding != 0:
        return result[:, :, :-self.__padding]
    return result

class TriangularCausalMask():
    def __init__(self, b, n, device="cpu"):
        mask_shape = [b, 1, n, n]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask

class MultiheadAttention(nn.Module):
  def __init__(self, d_model, n_heads, d_keys, mask_flag, r_att_drop=0.1):
    super(MultiheadAttention, self).__init__()
    self.h, self.d, self.mask_flag= n_heads, d_keys, mask_flag
    self.proj_q = nn.Linear(d_model, self.h * self.d)
    self.proj_k = nn.Linear(d_model, self.h * self.d)
    self.proj_v = nn.Linear(d_model, self.h * self.d)
    self.proj_out = nn.Linear(self.h * self.d, d_model)
    self.dropout = nn.Dropout(r_att_drop) 

  def forward(self, q, k, v):
    b, n_q, n_k, h, d = q.size(0), q.size(1), k.size(1), self.h, self.d

    q, k, v = self.proj_q(q), self.proj_k(k), self.proj_v(v)        # b, n_*, h*d
    q, k, v = map(lambda x: x.reshape(b, -1, h, d), [q, k, v])      # b, n_*, h, d
    scores = torch.einsum('bnhd,bmhd->bhnm', (q,k))                 # b, h, n_q, n_k
    
    if self.mask_flag:
      att_mask = TriangularCausalMask(b, n_q, device=q.device)
      scores.masked_fill_(att_mask.mask, -np.inf)

    att = F.softmax(scores / (self.d ** .5), dim=-1)                # b, h, n_q, n_k
    att = self.dropout(att)
    att_out = torch.einsum('bhnm,bmhd->bnhd', (att,v))              # b, n_q, h, d
    att_out = att_out.reshape(b, -1, h*d)                           # b, n_q, h*d
    out = self.proj_out(att_out)                                    # b, n_q, d_model
    return out
