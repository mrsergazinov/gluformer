import torch
import torch.nn as nn
import torch.nn.functional as F

from gluformer.attention import *

class DecoderLayer(nn.Module):
  def __init__(self, self_att, cross_att, d_model, d_fcn,
                r_drop, activ="relu"):
    super(DecoderLayer, self).__init__()

    self.self_att = self_att
    self.cross_att = cross_att

    self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_fcn, kernel_size=1)
    self.conv2 = nn.Conv1d(in_channels=d_fcn, out_channels=d_model, kernel_size=1)

    self.norm1 = nn.LayerNorm(d_model)
    self.norm2 = nn.LayerNorm(d_model)
    self.norm3 = nn.LayerNorm(d_model)

    self.dropout = nn.Dropout(r_drop)
    self.activ = F.relu if activ == "relu" else F.gelu

  def forward(self, x_dec, x_enc):
    x_dec = x_dec + self.self_att(x_dec, x_dec, x_dec)
    x_dec = self.norm1(x_dec)

    x_dec = x_dec + self.cross_att(x_dec, x_enc, x_enc)
    res = x_dec = self.norm2(x_dec)

    res = self.dropout(self.activ(self.conv1(res.transpose(-1,1))))
    res = self.dropout(self.conv2(res).transpose(-1,1))
    
    return self.norm3(x_dec+res)

class Decoder(nn.Module):
  def __init__(self, layers, norm_layer=None):
    super(Decoder, self).__init__()
    self.layers = nn.ModuleList(layers)
    self.norm = norm_layer

  def forward(self, x_dec, x_enc):
    for layer in self.layers:
      x_dec = layer(x_dec, x_enc)

    if self.norm is not None:
      x_dec = self.norm(x_dec)

    return x_dec
