import torch
import torch.nn as nn
import torch.nn.functional as F

from gluformer.attention import *

class ConvLayer(nn.Module):
  def __init__(self, d_model):
    super(ConvLayer, self).__init__()
    self.downConv = nn.Conv1d(in_channels=d_model, out_channels=d_model,
                              kernel_size=3, padding=1, padding_mode='circular')
    self.norm = nn.BatchNorm1d(d_model)
    self.activ = nn.ELU()
    self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

  def forward(self, x):
    x = self.downConv(x.transpose(-1, 1))
    x = self.norm(x)
    x = self.activ(x)
    x = self.maxPool(x)
    x = x.transpose(-1,1)
    return x

class EncoderLayer(nn.Module):
  def __init__(self, att, d_model, d_fcn, r_drop, activ="relu"):
    super(EncoderLayer, self).__init__()
    
    self.att = att
    self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_fcn, kernel_size=1)
    self.conv2 = nn.Conv1d(in_channels=d_fcn, out_channels=d_model, kernel_size=1)
    self.norm1 = nn.LayerNorm(d_model)
    self.norm2 = nn.LayerNorm(d_model)
    self.dropout = nn.Dropout(r_drop)
    self.activ = F.relu if activ == "relu" else F.gelu

  def forward(self, x):
    new_x = self.att(x, x, x)
    x = x + self.dropout(new_x)

    res = x = self.norm1(x)
    res = self.dropout(self.activ(self.conv1(res.transpose(-1,1))))
    res = self.dropout(self.conv2(res).transpose(-1,1))

    return self.norm2(x+res)

class Encoder(nn.Module):
  def __init__(self, enc_layers, conv_layers=None, norm_layer=None):
    super(Encoder, self).__init__()
    self.enc_layers = nn.ModuleList(enc_layers)
    self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
    self.norm = norm_layer

  def forward(self, x):
    # x [B, L, D]
    if self.conv_layers is not None:
        for enc_layer, conv_layer in zip(self.enc_layers, self.conv_layers):
            x = enc_layer(x)
            x = conv_layer(x)
        x = self.enc_layers[-1](x)
    else:
        for enc_layer in self.enc_layers:
            x = enc_layer(x)

    if self.norm is not None:
        x = self.norm(x)

    return x