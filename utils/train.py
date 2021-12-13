import numpy as np
import torch

from gluformer.model import *
from utils.collate import *

class EarlyStop:
  def __init__(self, patience, delta):
    self.patience = patience
    self.delta = delta
    self.counter = 0
    self.best_loss = np.Inf
    self.stop = False

  def __call__(self, loss, model, path):
    if loss < self.best_loss:
      print("Validation loss descreased: " + str(self.best_loss) + " -> " + str(loss))
      self.best_loss = loss
      self.counter = 0
      torch.save(model.state_dict(), path)
    elif loss > self.best_loss + self.delta:
      self.counter = self.counter + 1
      print("Validation loss did not decrease " + str(self.counter) + " / " + str(self.patience))
      if self.counter >= self.patience:
        self.stop = True

class ExpLikeliLoss(nn.Module):
  def __init__(self, num_samples = 100):
    super(ExpLikeliLoss, self).__init__()
    self.num_samples = num_samples

  def forward(self, pred, true):
    # pred & true: [b, l, d]
    SIGMA = 0.3
    b, l, d = pred.size(0), pred.size(1), pred.size(2)
    true = true.transpose(0,1).reshape(l, -1, self.num_samples).transpose(0, 1)
    pred = pred.transpose(0,1).reshape(l, -1, self.num_samples).transpose(0, 1)

    return torch.mean((-1) * torch.logsumexp(torch.sum((-1 / (2 * SIGMA)) * (true - pred) ** 2, dim=1), dim=1))
    
def modify_collate(num_samples):
  '''
  Repeat each sample in the dataset.
  '''
  def wrapper(batch):
    batch_rep = [sample for sample in batch for i in range(num_samples)]
    return default_collate(batch_rep)
  
  return wrapper

def adjust_learning_rate(model_optim, epoch, lr):
  lr = lr * (0.5 ** epoch)
  print("Learning rate halfing...")
  print(f"New lr: {lr:.7f}")
  for param_group in model_optim.param_groups:
    param_group['lr'] = lr

def process_batch(subj_id, 
                  batch_x, batch_y, 
                  batch_x_mark, batch_y_mark, 
                  len_pred, len_label, 
                  model, device):
  # read data
  subj_id = subj_id.long().to(device)
  batch_x = batch_x.float().to(device)
  batch_y = batch_y.float()
  batch_x_mark = batch_x_mark.float().to(device)
  batch_y_mark = batch_y_mark.float().to(device)
  
  # extract true
  true = batch_y[:, -len_pred:, :].to(device)

  # decoder input
  dec_inp = torch.zeros([batch_y.shape[0], len_pred, batch_y.shape[-1]]).float()
  dec_inp = torch.cat([batch_y[:, :len_label, :], dec_inp], dim=1).float().to(device)

  # model prediction
  pred = model(subj_id, batch_x, batch_x_mark, dec_inp, batch_y_mark)
  return pred, true