import os
import argparse
import time
import torch
from tqdm import tqdm

from lstm import lstm
from utils import mse_loss, dilate_loss, teacher_forcing_rate
from train_and_pred import train, pred
from dataset import get_data, get_state_diff_data, get_dataloader

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='2nd_version_data', help='root dir for training data')
    parser.add_argument('--file_name', type=str, default='mse_model', help='saved file name of model and log')
    parser.add_argument('--device', type=str, default='cuda', help='device for model training')

    parser.add_argument('--input_dim', type=int, default=21, help='input dimention for model')
    parser.add_argument('--output_dim', type=int, default=1, help='output dimention for model')
    parser.add_argument('--hidden_dim', type=int, default=64, help='hidden dimention for model')
    parser.add_argument('--unit_num', type=int, default=2, help='number of lstm units')

    parser.add_argument('--loss_type', type=str, default='mse', help='choose loss function, mse / dilate')
    parser.add_argument('--alpha', type=float, default=0.5, help='ratio of shape and temporol in dilate')
    parser.add_argument('--gamma', type=float, default=0.01, help='gamma for dilate loss')

    parser.add_argument('--max_epochs', type=int, default=1000, help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size for training')
    parser.add_argument('--lr', type=float,  default=0.001, help='learning rate for training')
    parser.add_argument('--weight_decay', type=float,  default=0.005, help='weight decay for training')
    parser.add_argument('--gtn', type=int, default=20, help='ground truth number for training and testing')
    parser.add_argument('--ratio', type=float, default=0.8, help='ratio of training set and validation set')

    parser.add_argument('--tfr_decay_rate', type=float, default=0.01, help='decay rate for teacher forcing rate')
    parser.add_argument('--tfr_lower_bound', type=float, default=0.0, help='lower bound for teacher forcing rate')

    args = parser.parse_args()
    return args

def get_log_string():
    log_string = '[{:03d}/{:03d}] {:2.2f} sec(s) Train Loss: {:3.4f} | Val loss: {:3.4f}'.format( \
        i_epoch, args.max_epochs, time.time()-epoch_start_time, train_loss/train_count, val_loss/val_count)
    return log_string

def update(progress):
    progress.set_description(get_log_string())
    progress.update(1)

if __name__ == '__main__':  

    # get args
    args = parse_args()
  
    # get training data
    X, Y, L = get_state_diff_data(args)

    # get dataloader
    train_loader, val_loader = get_dataloader(X, Y, L, args)

    # initail teacher forcing rate
    tfr = teacher_forcing_rate(args)

    # make dirs
    os.makedirs('./log', exist_ok=True)
    os.makedirs('./model', exist_ok=True)
    if os.path.exists(f'./log/{args.file_name}.txt'):
        os.remove(f'./log/{args.file_name}.txt')

    # initial model and optimizer
    model = lstm(args).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
  
    # choose loss function
    if args.loss_type == 'mse':
        loss_func = mse_loss
    elif args.loss_type == 'dilate':
        loss_func = dilate_loss
    else:
        print("Can't recognize loss function type.")

    # training loop
    best_val_loss = 1e9
    progress = tqdm(total = len(train_loader) + len(val_loader), \
        bar_format='{desc}{percentage:3.0f}%|{bar}|{n}/{total}[{rate_fmt}{postfix}]')
    for i_epoch in range(1, args.max_epochs):

        epoch_start_time = time.time()
        train_loss = 0.0
        train_count = 1e-8
        val_loss = 0.0
        val_count = 1e-8

        for x, y, l in train_loader:
            loss, count = train(model, optimizer, x, y, l, tfr, loss_func, args)
            train_loss += loss
            train_count += count
            update(progress)

        for x, y, l in val_loader:
            loss, count = pred(model, x, y, l, loss_func, args)
            val_loss += loss
            val_count += count
            update(progress)

        if val_loss/val_count < best_val_loss:
            best_val_loss = val_loss/val_count
            torch.save(model.state_dict(), f'./model/{args.file_name}.pth')

        progress.reset()
        with open(f'./log/{args.file_name}.txt', 'a') as train_record:
            train_record.write(get_log_string() + '\n')
      