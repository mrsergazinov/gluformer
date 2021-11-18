import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEmbedding(nn.Module):
  def __init__(self, d_model, max_len=5000):
    super(PositionalEmbedding, self).__init__()
    # Compute the positional encodings once in log space.
    pos_emb = torch.zeros(max_len, d_model).float()
    pos_emb.require_grad = False

    position = torch.arange(0, max_len).float().unsqueeze(1)
    div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

    pos_emb[:, 0::2] = torch.sin(position * div_term)
    pos_emb[:, 1::2] = torch.cos(position * div_term)

    pos_emb = pos_emb.unsqueeze(0)
    self.register_buffer('pos_emb', pos_emb)

  def forward(self, x):
    return self.pos_emb[:, :x.size(1)]

class TokenEmbedding(nn.Module):
  def __init__(self, d_model):
    super(TokenEmbedding, self).__init__()
    D_INP = 1 # one sequence
    self.conv = nn.Conv1d(in_channels=D_INP, out_channels=d_model, 
                          kernel_size=3, padding=1, padding_mode='circular')
    # nn.init.kaiming_normal_(self.conv.weight, mode='fan_in', nonlinearity='leaky_relu')

  def forward(self, x):
    x = self.conv(x.transpose(-1, 1)).transpose(-1, 1)
    return x

class TemporalEmbedding(nn.Module):
  def __init__(self, d_model):
    super(TemporalEmbedding, self).__init__()
    NUM_FEATURES = 5
    self.embed = nn.Linear(NUM_FEATURES, d_model)
  
  def forward(self, x):
    x = x.float()
    return self.embed(x)

class SubjectEmbedding(nn.Module):
  def __init__(self, d_model):
    super(SubjectEmbedding, self).__init__()
    self.id_embedding = nn.Linear(1, d_model)

  def forward(self, x):
    x = x.float().unsqueeze(1)
    embed_x = self.id_embedding(x)

    return embed_x

class DataEmbedding(nn.Module):
  def __init__(self, d_model, r_drop):
    super(DataEmbedding, self).__init__()
    # note: d_model // 2 == 0
    self.value_embedding = TokenEmbedding(d_model)
    self.time_embedding = TemporalEmbedding(d_model) # alternative: TimeFeatureEmbedding
    self.positional_embedding = PositionalEmbedding(d_model)
    self.subject_embedding = SubjectEmbedding(d_model)
    self.dropout = nn.Dropout(r_drop)

  def forward(self, x_id, x, x_mark):
    x =  self.value_embedding(x) + self.positional_embedding(x) + self.time_embedding(x_mark)
    x = torch.cat((self.subject_embedding(x_id).unsqueeze(1), x), dim = 1)
    return self.dropout(x)
