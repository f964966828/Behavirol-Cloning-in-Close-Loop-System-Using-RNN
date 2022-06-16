import os
from numba import jit
import torch
import torch.nn as nn

class teacher_forcing_rate():
    def __init__(self, args):
        self.tfr = 1
        self.tfr_decay_rate = args.tfr_decay_rate
        self.tfr_lower_bound = args.tfr_lower_bound
    
    def update(self):
        self.tfr = max(self.tfr - self.tfr_decay_rate, self.tfr_lower_bound)
  
    def get_tfr(self):
        self.update()
        return self.tfr

def pairwise_distances(x, y):
    '''
    Input: x is a Nxd matrix
           y is an optional Nxd matirx
    Output: D is a NxM matrix where D[i,j] is the square norm between x[i,:] and y[j,:]
    i.e. D[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x**2).sum(1).view(-1, 1)

    y_t = torch.transpose(y, 0, 1)
    y_norm = (y**2).sum(1).view(1, -1)

    # (x-y)^2 = x^2 + y^2 - 2xy
    D = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return torch.clamp(D, 0.0, float('inf'))

def pairwise_penalty(N):
    '''
    Input: x is a Nxd matrix
           y is an optional Nxd matirx
    Output: P is a NxM matrix where P[i,j] is the penalty between i and j
    i.e. P[i,j] = (i - j)^2 / n^2
    '''
    matrix1 = torch.tensor([[i for _ in range(N)] for i in range(N)])
    matrix2 = torch.tensor([[j for j in range(N)] for _ in range(N)])
    
    P = (matrix1-matrix2)**2 / N**2
    return P

class mse_loss():
    def __init__(self, y_true, y_pred, length, args):
        self.y_true = y_true
        self.y_pred = y_pred
        self.length = length

    def calculate_loss(self):
        loss = 0.0
        count = 0.0
        for (seq1, seq2, l) in zip(self.y_true, self.y_pred, self.length):
            self.D = pairwise_distances(seq1, seq2)
            for i in range(l):
                loss += self.D[i, i]
            count += l
        return loss, count

class dilate_loss():
    def __init__(self, y_true, y_pred, length, args):
        self.y_true = y_true
        self.y_pred = y_pred
        self.length = length
        self.alpha = args.alpha
        self.gamma = args.gamma

    def calculate_loss(self):
        # TODO
        pass