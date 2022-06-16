import random
import numpy as np
import torch

def train(model, optimizer, x, y, l, tfr, loss_func, args):
    model.zero_grad()
    model.hidden = model.init_hidden(x.shape[0])
    y = y.reshape(y.shape[0], y.shape[1], args.output_dim)
    
    y_pred = y[:, 0]
    pred_list = [y_pred.reshape(y.shape[0], 1, args.output_dim)]
    
    use_teacher_forcing = True if random.random() < tfr.get_tfr() else False
    
    for i in range(1, x.shape[1]):
        if use_teacher_forcing or i < args.gtn:
            y_pred = model(torch.cat([x[:, i], y[:, i-1]], 1))
        else:
            y_pred = model(torch.cat([x[:, i], y_pred], 1))
        pred_list.append(y_pred.reshape(y.shape[0], 1, args.output_dim))
    
    y_pred = torch.cat(pred_list, 1)
    loss = loss_func(y, y_pred, l, args)
    batch_loss, count = loss.calculate_loss()

    batch_loss.backward()
    optimizer.step()
    
    return batch_loss.item(), count

def pred(model, x, y, l, loss_func, args):
    model.hidden = model.init_hidden(x.shape[0])
    y = y.reshape(y.shape[0], y.shape[1], args.output_dim)
    
    loss = 0.0
    count = 0.0
    y_pred = y[:, 0]
    pred_list = [y_pred.reshape(y.shape[0], 1, args.output_dim)]
    
    for i in range(1, x.shape[1]):
        if i < args.gtn:
            y_pred = model(torch.cat([x[:, i], y[:, i-1]], 1))
        else:
            y_pred = model(torch.cat([x[:, i], y_pred], 1))
        pred_list.append(y_pred.reshape(y.shape[0], 1, args.output_dim))
    
    y_pred = torch.cat(pred_list, 1)
    loss = loss_func(y, y_pred, l, args)
    batch_loss, count = loss.calculate_loss()

    return batch_loss.item(), count