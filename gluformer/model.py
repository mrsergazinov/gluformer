import torch
import torch.nn as nn
import torch.nn.functional as F

from gluformer.embed import *
from gluformer.attention import *
from gluformer.encoder import *
from gluformer.decoder import *
from gluformer.variance import *

class Gluformer(nn.Module):
  def __init__(self, d_model, n_heads, d_fcn, r_drop, 
                activ, num_enc_layers, num_dec_layers, 
                distil, len_seq, len_pred, num_features=5):
    super(Gluformer, self).__init__()
    # Set prediction length
    self.len_pred = len_pred
    # Embedding
    # note: d_model // 2 == 0
    self.enc_embedding = DataEmbedding(d_model, r_drop, num_features)
    self.dec_embedding = DataEmbedding(d_model, r_drop, num_features)
    # Encoding
    self.encoder = Encoder(
      [
        EncoderLayer(
          att=MultiheadAttention(d_model=d_model, n_heads=n_heads, 
                                  d_keys=d_model//n_heads, mask_flag=False, 
                                  r_att_drop=r_drop),
          d_model=d_model,
          d_fcn=d_fcn,
          r_drop=r_drop,
          activ=activ) for l in range(num_enc_layers)
      ],
      [
        ConvLayer(
          d_model) for l in range(num_enc_layers-1)
      ] if distil else None, 
      norm_layer=torch.nn.LayerNorm(d_model)
    )

    # Decoding
    self.decoder = Decoder(
      [
        DecoderLayer(
          self_att=MultiheadAttention(d_model=d_model, n_heads=n_heads, 
                                  d_keys=d_model//n_heads, mask_flag=True, 
                                  r_att_drop=r_drop),
          cross_att=MultiheadAttention(d_model=d_model, n_heads=n_heads, 
                                  d_keys=d_model//n_heads, mask_flag=False, 
                                  r_att_drop=r_drop),
          d_model=d_model,
          d_fcn=d_fcn,
          r_drop=r_drop,
          activ=activ) for l in range(num_dec_layers)
      ], 
      norm_layer=torch.nn.LayerNorm(d_model)
    )
    
    # Output
    D_OUT = 1
    self.projection = nn.Linear(d_model, D_OUT, bias=True)

    # Train variance
    self.var = Variance(d_model, r_drop, len_seq)

  def forward(self, x_id, x_enc, x_mark_enc, x_dec, x_mark_dec):
    enc_out = self.enc_embedding(x_id, x_enc, x_mark_enc)
    var_out = self.var(enc_out)
    enc_out = self.encoder(enc_out)

    dec_out = self.dec_embedding(x_id, x_dec, x_mark_dec)
    dec_out = self.decoder(dec_out, enc_out)
    dec_out = self.projection(dec_out)
    
    return dec_out[:, -self.len_pred:, :], var_out # [B, L, D], log variance



