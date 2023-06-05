# -*- coding: utf-8 -*-
import os

import sys
sys.path.append("../")

import time
import random
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import utils as utl
from models.nets import SSL_MODEL
from dataloaders.Datasets import PresageDataset
import signal_transformation.signal_transformation_task as stt

torch.cuda.empty_cache()

SEED = 1
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


def Train(model, iterator, CUDA, device, criterion_c, optimizer):
    epoch_loss = 0
    epoch_acc = 0

    epoch_loss_eda = 0
    epoch_loss_bvp = 0
    epoch_loss_temp = 0

    epoch_acc_eda = 0
    epoch_acc_bvp = 0
    epoch_acc_temp = 0

    # put model into train mode
    model.train()
    for i_batch, sample_batched in enumerate(iterator):
        
        trans, y = apply_trans(sample_batched)
        trans = torch.tensor(trans, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.long)
    
        if CUDA:
            trans = trans.to(device)
            y = y.to(device)

        
        optimizer.zero_grad()
        
        eda_c, bvp_c, temp_c = model(trans[:,:,0], trans[:,:,1], trans[:,:,2])

        loss_eda = criterion_c(eda_c, y)
        loss_bvp = criterion_c(bvp_c, y)
        loss_temp = criterion_c(temp_c, y)

        loss = loss_eda + loss_bvp + loss_temp 
        
        loss.backward()
        optimizer.step()
        
        eda_pred = torch.max(eda_c.data, 1)[1]
        acc_eda = utl.calculate_accuracy(eda_pred, y)
        
        bvp_pred = torch.max(bvp_c.data, 1)[1]
        acc_bvp = utl.calculate_accuracy(bvp_pred, y)
        
        temp_pred = torch.max(temp_c.data, 1)[1]
        acc_temp = utl.calculate_accuracy(temp_pred, y)
        
        acc = (acc_eda + acc_bvp + acc_temp)/3
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
        epoch_loss_eda += loss_eda.item()
        epoch_loss_bvp += loss_bvp.item()
        epoch_loss_temp += loss_temp.item()

        epoch_acc_eda += acc_eda.item()
        epoch_acc_bvp += acc_bvp.item()
        epoch_acc_temp += acc_temp.item()


    loss_list = [epoch_loss_eda, epoch_loss_bvp, epoch_loss_temp, epoch_loss]
    acc_list = [epoch_acc_eda, epoch_acc_bvp, epoch_acc_temp, epoch_acc]

    
    return [x / len(iterator) for x in loss_list], [x / len(iterator) for x in acc_list]


def apply_trans(batch_data):
    trans_list = []
    choice_list = []
    batch_data_new = batch_data.detach().numpy().copy()
    for sample_idx in range(len(batch_data_new)):
        sample = batch_data_new[sample_idx,:,:]
        trans_arr = np.zeros((6, batch_data_new.shape[1], batch_data_new.shape[-1]))
        for i in range(sample.shape[-1]):
            noised = stt.add_noise_with_SNR(sample[:,i], noise_amount =15)
            permuted = stt.permute(sample[:,i], pieces = 10)
            time_wraped = stt.time_warp_v3(sample[:,i], sampling_freq = 4,
                                                          pieces = 9, stretch_factor =1.05,
                                                          squeeze_factor = 1/1.05)
            crop_resized = stt.CropResize(sample[:,i], nPerm =4)
            MagWarpped = stt.MagWarp(sample[:,i], sigma = 0.2)
            trans_arr[:,:,i] =  np.array([sample[:,i], noised, permuted, time_wraped, crop_resized, MagWarpped])

        trans_list.append(trans_arr)
        choice_list.append(np.array([0,1,2,3,4,5]))

    trans = np.concatenate(trans_list)
    choices = np.concatenate(choice_list)

    return trans, choices


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='Path to SSL files', type=str, required=True)
    parser.add_argument('--data_path', help='Path to dataset', type=str, required=True)
    
    parser.add_argument('--num_epoch_SSL', help='Number of epochs for self-supervised learning', type=int, default=15)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=32)
    parser.add_argument('--optim', help='Optimizer', type=str, default='sgd')
    parser.add_argument('--lr', help='Learning rate', type=float, default=5e-3)
    parser.add_argument('--momentum', help='Momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', help='Weight decay', type=float, default=5e-7)
    parser.add_argument('--use_cuda', help='CUDA option', type=str, default=True)
    
    parser.add_argument('--tcn_nfilters', nargs='+', help='Number of TCN filters', default=[16, 16])
    parser.add_argument('--tcn_kernel_size', help='TCN filter size', type=int, default=6)
    parser.add_argument('--tcn_dropout', help='Dropout in TCN', type=float, default=0.2)
    parser.add_argument('--trans_d_model', help='Dimension of input embedding in Transformer', type=int, default=128)
    parser.add_argument('--trans_n_heads', help='Number of heads in Transformer', type=int, default=4)
    parser.add_argument('--trans_num_layers', help='Number of Encoderlayers', type=int, default=1)
    parser.add_argument('--trans_dim_feedforward', help='Dimension of the feedforward network', type=int, default=128)
    parser.add_argument('--shared_embed_dim', type=int, default=64)
    parser.add_argument('--trans_dropout', help='Dropout in Transformer', type=float, default=0.2)
    parser.add_argument('--trans_activation', help='Activation in Transformer', type=str, default='relu')
    parser.add_argument('--trans_norm', help='Normalizarion in Transformer', type=str, default='LayerNorm')
    parser.add_argument('--ssl_embed_dim', help='Dimension of the feedforward network for classification', type=int, default=64)
    parser.add_argument('--ssl_num_classes', help='Number of transformations', type=int, default=6)
    parser.add_argument('--ssl_activation', help='Activation in the classifier', type=str, default='relu')
    parser.add_argument('--ssl_dropout', help='Dropout in the classifier', type=float, default=0.1)
    
    
    args = parser.parse_args()
    return args


def main(args):

    current_t = utl.current_time()
    
    # log path  
    output_ssl_dir = os.path.join(args.path, "log_SSL", current_t)
    utl.create_dir(output_ssl_dir)
        
    # Pandas dataframe for log file
    df_log_ssl = pd.DataFrame(columns = ["epoch", 
                                          "Train_loss_eda", "Train_loss_bvp", 
                                          "Train_loss_temp", "Train_loss_total",
                                          "Train_acc_eda", "Train_acc_bvp",
                                          "Train_acc_temp", "Train_acc_total"]) 
    
    # model path(check point and best)  
    checkpoint_dir = os.path.join(args.path, "saved_model", "SSL", current_t, "ckpoint")
    utl.create_dir(checkpoint_dir)
    best_model_dir = os.path.join(args.path, "saved_model", "SSL",current_t,"best")
    utl.create_dir(best_model_dir)
        
    ############################## Load Presage dataset ###########################
    
    presage_X = utl.load_presage(args.data_path)
    train = PresageDataset(presage_X)
    iter_train = DataLoader(train, batch_size = args.batch_size, shuffle=True)
    
    ########################### Define SSL Model ##################################
    
    model = SSL_MODEL(tcn_nfilters = args.tcn_nfilters,
              tcn_kernel_size = args.tcn_kernel_size,
              tcn_dropout = args.tcn_dropout, 
              trans_d_model = args.trans_d_model, 
              trans_n_heads = args.trans_n_heads, 
              trans_num_layers = args.trans_num_layers, 
              trans_dim_feedforward = args.trans_dim_feedforward, 
              shared_embed_dim = args.shared_embed_dim,
              trans_dropout = args.trans_dropout, 
              trans_activation = args.trans_activation, 
              trans_norm = args.trans_norm, 
              trans_freeze = False,
              ssl_embed_dim = args.ssl_embed_dim, 
              ssl_num_classes = args.ssl_num_classes, 
              ssl_activation = args.ssl_activation, 
              ssl_dropout = args.ssl_dropout)
    
    ########################### Define Loss Function ##############################
    
    criterion = nn.CrossEntropyLoss()
    
    ########################### Set up the optimizer ############################## 
    
    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                weight_decay=args.weight_decay,
                                betas=(0.95, 0.999))
    else:
        raise ValueError('Optimizer %s is not supported' % args.optim)
    
    ########################### Set up CUDA #######################################        
    
    device = None
    if args.use_cuda:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model = model.to(device)
        criterion = criterion.to(device)
    
    ###################### Self-supervised learning ###############################
    
    best_acc = -np.inf
    
    for epoch in range(args.num_epoch_SSL):
        start_time = time.time()
        train_loss, train_acc  = Train(model, 
                                      iter_train, 
                                      args.use_cuda, 
                                      device, 
                                      criterion, 
                                      optimizer)
        end_time = time.time() 
        epoch_mins, epoch_secs = utl.epoch_time(start_time, end_time)
        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss[-1]:.3f} | Train Acc: {train_acc[-1]*100:.2f}%')
        df_log_ssl = df_log_ssl.append({
                                        "epoch":epoch+1,
                                        "Train_loss_eda":train_loss[0], 
                                        "Train_loss_bvp":train_loss[1],
                                        "Train_loss_temp":train_loss[2],
                                        "Train_loss_total":train_loss[-1],
                                        "Train_acc_eda":train_acc[0]*100,
                                        "Train_acc_bvp":train_acc[1]*100,
                                        "Train_acc_temp":train_acc[2]*100,
                                        "Train_acc_total":train_acc[-1]*100
                                        }, ignore_index=True)
        
        state_ckp = {
            'epoch': epoch + 1,
            'state_dict_fusion': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }  
        
        utl.save_ckp(state_ckp, checkpoint_dir, epoch)
        utl.save_best(state_ckp, train_acc[-1] > best_acc, best_model_dir)
        best_acc = max(best_acc, train_acc[-1])
    
    df_log_ssl.to_csv(os.path.join(output_ssl_dir, "SSL_train.csv"), index=False)  


if __name__ == "__main__":   
    args = parse_args()   
    main(args)


