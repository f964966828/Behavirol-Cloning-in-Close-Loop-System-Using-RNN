import math
import torch
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn.utils.rnn as rnn_utils

def get_data(args):
    X, Y = list(), list()
    for file in os.listdir(args.data_root):
        df = pd.read_excel(f'{args.data_root}/{file}')
        X.append(np.array(df.drop(['time', 'propofol'], axis=1)).astype(np.float32))
        Y.append(np.array(df['propofol']).astype(np.float32))

    X = [torch.tensor(x) for x in X]
    Y = [torch.tensor(y) for y in Y]
    L = [len(x) for x in X]
    
    X = rnn_utils.pad_sequence(X, batch_first=True)
    Y = rnn_utils.pad_sequence(Y, batch_first=True)
    
    print(f'Total {len(X)} of data has been loaded.')
    return X, Y, L

def get_state_diff_data(args):
    X, Y = list(), list()
    for file in os.listdir(args.data_root)[:10]:
    # for file in os.listdir(args.data_root):
        df = pd.read_excel(f'{args.data_root}/{file}')
        df = df.drop(['time'], axis=1)
        column_name = df.columns
    
        for col in column_name:
      
            # propofol is our model output
            if col == 'propofol':
                continue
                
            tmp = [0.0]
            for i in range(1, len(df[col])):
                tmp.append(df[col][i] - df[col][i-1])
            new_col = col + '_diff'
            df[new_col] = tmp
      
        X.append(np.array(df.drop(['propofol'], axis=1)).astype(np.float32))
        Y.append(np.array(df['propofol']).astype(np.float32))

    X = [torch.tensor(x) for x in X]
    Y = [torch.tensor(y) for y in Y]
    L = [len(x) for x in X]

    # Let all data be equal length with padding zeros
    # L is the lengtg list that store the origin length
    X = rnn_utils.pad_sequence(X, batch_first=True)
    Y = rnn_utils.pad_sequence(Y, batch_first=True)

    print(f'Total {len(X)} of data has been loaded.')
    return X, Y, L

class RNN_Dataset(Dataset):
    def __init__(self, X, Y, L, args):
        self.X = X
        self.Y = Y
        self.L = L
        self.device = args.device
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        x = self.X[index].to(self.device)
        y = self.Y[index].to(self.device)
        l = self.L[index]
        return x, y, l

def get_dataloader(X, Y, L, args):
    train_x, val_x = X[:math.ceil(len(X)*args.ratio)], X[math.ceil(len(Y)*args.ratio):]
    train_y, val_y = Y[:math.ceil(len(X)*args.ratio)], Y[math.ceil(len(Y)*args.ratio):]
    train_l, val_l = L[:math.ceil(len(X)*args.ratio)], L[math.ceil(len(Y)*args.ratio):]

    train_set = RNN_Dataset(train_x, train_y, train_l, args)
    val_set = RNN_Dataset(val_x, val_y, val_l, args)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    print(f'training size: {sum(train_l)}, validation size: {sum(val_l)}')

    return train_loader, val_loader