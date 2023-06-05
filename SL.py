# -*- coding: utf-8 -*-


import os
import sys
import time
import random
import argparse
import numpy as np
import pandas as pd
sys.path.append("../")


import torch
import torch.nn as nn
from torch.utils.data import DataLoader


import utils as utl
from models.nets import SL_model
from dataloaders.Datasets import SupervisedDatasets

torch.cuda.empty_cache()
SEED = 1
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


def Train_SL(model_sl, iterator, CUDA, device, criterion, optimizer):
    epoch_loss = 0
    epoch_acc = 0
    
    model_sl.train()
        
    for i_batch, sample_batched in enumerate(iterator):

        eda = sample_batched["eda"].type(torch.float32)
        bvp = sample_batched["bvp"].type(torch.float32)
        temp = sample_batched["temp"].type(torch.float32)
        y = sample_batched["label"].type(torch.LongTensor)
        
        if CUDA:
            eda = eda.to(device)
            bvp = bvp.to(device)
            temp = temp.to(device)
            y = y.to(device)
        
        optimizer.zero_grad()
        
        output = model_sl(eda, bvp, temp)
        y_pred = torch.max(output.data, 1)[1]

        if np.size(np.array(y.cpu())) == 1:
            loss = criterion(output, y)
            
        else:
           
            loss = criterion(output, y.squeeze())

        
   
        loss.backward()
        optimizer.step()
        
        acc = utl.calculate_accuracy(y_pred, y)
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()

    
    return epoch_loss / len(iterator) , epoch_acc/ len(iterator)

def Evaluate_SL(model_sl, iterator, CUDA, device, criterion, num_classes):
    epoch_loss = 0
    epoch_acc = 0
    
    model_sl.eval()
        
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(iterator):
     
            eda = sample_batched["eda"].type(torch.float32)
            bvp = sample_batched["bvp"].type(torch.float32)
            temp = sample_batched["temp"].type(torch.float32)
            y = sample_batched["label"].type(torch.LongTensor)
    
            if CUDA:
                eda = eda.to(device)
                bvp = bvp.to(device)
                temp = temp.to(device)
                y = y.to(device)
                
            output = model_sl(eda, bvp, temp)
            y_pred = torch.max(output.data, 1)[1]

            if np.size(np.array(y.cpu())) == 1:
                loss = criterion(output, y.squeeze(0))
            else:

                loss = criterion(output, y.squeeze())
                
            acc = utl.calculate_accuracy(y_pred, y)
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()


    return epoch_loss / len(iterator) , epoch_acc/ len(iterator)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='Path to SL files', type=str, required=True)
    parser.add_argument('--dataset_opt', help='Option for dataset', type=str, required=True)
    parser.add_argument('--data_path', help='Path to dataset', type=str, required=True)
    parser.add_argument('--best_model_dir', help='Path to pretrained models', type=str, required=True)
    parser.add_argument('--sl_num_classes', help='Number of emotions', type=int, required=True)
    parser.add_argument('--mode', help='Training mode', type=str, required=True, default='freeze')
    parser.add_argument('--av_opt', help='arousal/valence for CASE and KemoCon datasets', type=str, default='valence')
    
    parser.add_argument('--num_epoch_SL_w', help='Number of epochs for wesad dataset', type=int, default=20)
    parser.add_argument('--batch_size_SL_w', help='Batch size', type=int, default=128)
    parser.add_argument('--lr_w', help='Learning rate', type=float, default= 1e-4)
    
    parser.add_argument('--num_epoch_SL_ck', help='Number of epochs for case/kemocon dataset', type=int, default=64)
    parser.add_argument('--batch_size_SL_ck', help='Batch size', type=int, default=64)
    parser.add_argument('--lr_ck', help='Learning rate', type=float, default= 1e-3)
    
    parser.add_argument('--optim', help='Optimizer', type=str, default='sgd')
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
    parser.add_argument('--sl_embed_dim1', help='Dimension of the first FC layer for classification', type=int, default=128)
    parser.add_argument('--sl_embed_dim2', help='Dimension of the second FC layer for classification', type=int, default=64)
    parser.add_argument('--sl_activation', help='Activation in the classifier', type=str, default='relu')
    parser.add_argument('--sl_dropout', help='Dropout in the classifier', type=float, default=0.2)
    
    
    args = parser.parse_args()
    return args



def main(args):
    current_t = utl.current_time()
    
    # Load the selected dataset
    if args.dataset_opt == "WESAD":
        X, y, suj, cv_suj = utl.load_wesad(args.data_path)
        batch_size_SL = args.batch_size_SL_w
        num_epoch_SL = args.num_epoch_SL_w
        lr = args.lr_w
        
        # log path  
        output_sl_dir = os.path.join(args.path, "log_SL", args.dataset_opt, 
                                     str(args.sl_num_classes), current_t)
        utl.create_dir(output_sl_dir)
        # Pandas dataframe for log file
        df_log_sl = pd.DataFrame(columns = ["mode","fold","epoch",
                                            "Train_loss","Train_acc",
                                            "Val_loss","Val_acc"]) 
    
    if args.dataset_opt in ["CASE", "KemoCon"]:
        X, y, suj, cv_suj = utl.load_case_kemocon(args.data_path, args.av_opt, 
                                                  args.sl_num_classes)
        batch_size_SL = args.batch_size_SL_ck
        num_epoch_SL = args.num_epoch_SL_ck
        lr = args.lr_ck
        
        # log path
        output_sl_dir = os.path.join(args.path, "log_SL", args.dataset_opt, 
                                     args.av_opt, str(args.sl_num_classes), 
                                     current_t)
        utl.create_dir(output_sl_dir)
        # Pandas dataframe for log file
        df_log_sl = pd.DataFrame(columns = ["mode","fold","epoch",
                                            "Train_loss","Train_acc",
                                            "Val_loss","Val_acc"]) 

    ################Leave-one-subject-out cross validation ####################
    for fold_i in range(len(cv_suj)):  
        print(f'-----------Fold {fold_i+1}-----------')
        
        X_train, X_valid, y_train, y_valid = utl.fold_data_sl(X, y, cv_suj, suj, fold_i)
        
        if args.dataset_opt == "WESAD":
            X_train, y_train, X_valid, y_valid = utl.select_data(args.sl_num_classes, 
                                                                 X_train, X_valid, 
                                                                 y_train, y_valid)
            train = SupervisedDatasets(X_train, y_train-1)
            valid = SupervisedDatasets(X_valid, y_valid-1)
            
        else:
            train = SupervisedDatasets(X_train, y_train)
            valid = SupervisedDatasets(X_valid, y_valid)
            
        iter_train = DataLoader(train, batch_size = batch_size_SL, shuffle=True)
        iter_valid = DataLoader(valid, batch_size = batch_size_SL)

    ########################### Define SL Model ############################### 
        model_sl = SL_model(best_model_dir = args.best_model_dir,
                            SL_option = args.mode, 
                            num_classes = args.sl_num_classes, 
                            CUDA = args.use_cuda,
                            tcn_nfilters = args.tcn_nfilters,
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
                            sl_embed_dim1 = args.sl_embed_dim1, 
                            sl_embed_dim2 = args.sl_embed_dim2,
                            sl_activation = args.sl_activation, 
                            sl_dropout = args.sl_dropout)
        
        # # Verify number of trainable parameters
        # total_params = sum(p.numel() for p in model_sl.parameters() if p.requires_grad)
        
    ############################### Define Loss Function ######################
        criterion = nn.CrossEntropyLoss()
        
    ########################### Set up the optimizer ##########################
        if args.optim == 'sgd':
            optimizer = torch.optim.SGD(model_sl.parameters(), lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        elif args.optim == 'adam':
            optimizer = torch.optim.Adam(model_sl.parameters(), lr,
                                    weight_decay=args.weight_decay,
                                    betas=(0.95, 0.999))
        else:
            raise ValueError('Optimizer %s is not supported' % args.optim)    
            
    ########################### Set up CUDA ###################################      
        device = None
        if args.use_cuda:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            model_sl = model_sl.to(device)
            criterion = criterion.to(device)      
            
    ############################ Supervised Learning ##########################    
        for epoch in range(num_epoch_SL):
            start_time = time.time()
    
            train_loss, train_acc = Train_SL(model_sl, iter_train, 
                                              args.use_cuda, device, 
                                              criterion, optimizer)
            
            valid_loss, valid_acc = Evaluate_SL(model_sl, iter_valid, 
                                                args.use_cuda, device, 
                                                criterion, args.sl_num_classes)
          
    
            end_time = time.time() 
            epoch_mins, epoch_secs = utl.epoch_time(start_time, end_time)
            # print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            # print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
            # print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
            
            df_log_sl = df_log_sl.append({"mode":args.mode,
                                          "fold":fold_i,
                                          "epoch":epoch+1,
                                          "Train_loss":train_loss,
                                          "Train_acc":train_acc*100,
                                          "Val_loss":valid_loss,
                                          "Val_acc":valid_acc*100}, ignore_index=True)



    df_log_sl.to_csv(os.path.join(output_sl_dir, "mode_"+ args.mode +".csv"), index=False)      
    metric_df = df_log_sl[df_log_sl["mode"]== args.mode].groupby(['fold']).max().mean()
    print(metric_df)   
            
if __name__ == "__main__":   
    args = parse_args()   
    main(args)        
        
        
