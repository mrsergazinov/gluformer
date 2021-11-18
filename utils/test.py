import numpy as np
import torch

from gluformer.model import *

def predict_batch(subj_id, 
                  batch_x, 
                  batch_y, 
                  batch_x_mark, 
                  batch_y_mark, 
                  len_pred, 
                  len_pred_model, 
                  len_label, 
                  model, 
                  device):       
  # assume:
  # len_pred >= len_pred_model
  # len_label >= len_pred

  # read data
  subj_id = subj_id.long().to(device)
  batch_x = batch_x.float().to(device)
  batch_y = batch_y.float().to(device)
  batch_x_mark = batch_x_mark.float().to(device)
  batch_y_mark = batch_y_mark.float().to(device)

  # extract true
  true = batch_y[:, -len_pred:, :].to(device)

  # decoder input
  zeros = torch.zeros([batch_y.shape[0], len_pred_model, batch_y.shape[-1]]).float().to(device)
  dec_inp = torch.cat([batch_y[:, :len_label, :], zeros], dim=1).float().to(device)
  pred = model(subj_id, batch_x, batch_x_mark, dec_inp, batch_y_mark[:, :(len_label+len_pred_model), :])

  for i in range(1, len_pred // len_pred_model):
    dec_inp = torch.cat([batch_y[:, (i*len_pred_model):len_label, :], pred, zeros], dim=1).float().to(device)
    dec_inp_mark = batch_y_mark[:, (i*len_pred_model):(len_label+(i+1)*len_pred_model), :].float().to(device)
    enc_inp = torch.cat([batch_x[:, (i*len_pred_model):, :], pred], dim=1).float().to(device)
    enc_inp_mark = torch.cat([batch_x_mark[:, (i*len_pred_model):, :], 
                              batch_y_mark[:, len_label:(len_label+i*len_pred_model), :]], dim=1).float().to(device)

    pred_step = model(subj_id, 
                      enc_inp, 
                      enc_inp_mark, 
                      dec_inp, 
                      dec_inp_mark)
    pred = torch.cat([pred, pred_step], dim=1).float().to(device)

  return pred, true