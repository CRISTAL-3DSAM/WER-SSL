# -*- coding: utf-8 -*-


import os
import time
import torch
import joblib
import numpy as np
from sklearn.model_selection import KFold

def current_time():
    """ taking the current system time"""
    cur_time = time.strftime("%Y_%m_%d_%H_%M", time.gmtime())
    return cur_time

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def calculate_accuracy(y_pred, y):
    correct = y_pred.eq(y.data.view_as(y_pred)).long().cpu().sum()
    acc = correct.float() / y.shape[0]
    return acc

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    
def save_ckp(state, checkpoint_dir, epoch):
    f_path = os.path.join(checkpoint_dir, "ckp_model_epoch_" + str(epoch) +".pth")
    torch.save(state, f_path)
    
def save_best(state, save_flag, best_model_dir):
    f_path = os.path.join(best_model_dir, "best_model.pth")
    if save_flag:
        torch.save(state, f_path)

def load_presage(data_path):
    X = []
    scenario = ["preccc_v2_augmented_995", "focus_v2_augmented_995", 
                "newpassation_augmented_995", "rea_augmented_995"]
    for sub_dir in os.listdir(data_path):
        if sub_dir in scenario:
            sub_path = os.path.join(data_path, sub_dir, "time")
            for presage_file in os.listdir(sub_path):
                presage_data = joblib.load(os.path.join(sub_path, presage_file))
                X.append(presage_data["segs"])
    
    X = np.concatenate(X)
    return X

def load_wesad(data_path):
    wesad_X = []
    wesad_y = []
    wesad_suj = []
    for wesad_file in os.listdir(data_path):
        wesad_data = joblib.load(os.path.join(data_path, wesad_file))
        wesad_X.append(wesad_data["segs"])
        wesad_y.append(wesad_data["labels"])
        wesad_suj.append(np.array([int(wesad_file.split('.')[0])]*len(wesad_data["labels"])))
        
    wesad_X = np.concatenate(wesad_X)
    wesad_y = np.concatenate(wesad_y)
    wesad_suj = np.concatenate(wesad_suj)
    
     
    cv_wesad = KFold(n_splits = len(np.unique(wesad_suj)),  shuffle=False)
    cv_suj_wesad = list(cv_wesad.split(np.unique(wesad_suj)))
    
    return wesad_X, wesad_y, wesad_suj, cv_suj_wesad


def fold_data_sl(X, y, cv, suj_array, fold):
    suj_tr, suj_te = cv[fold]
    tr_id = np.unique(suj_array)[suj_tr]
    te_id = np.unique(suj_array)[suj_te]
    fold_X_train = X[np.isin(suj_array, tr_id)]
    fold_X_test = X[np.isin(suj_array, te_id)]
    
    fold_y_train = y[np.isin(suj_array, tr_id)]
    fold_y_test =y[np.isin(suj_array, te_id)]
    
    return fold_X_train, fold_X_test, fold_y_train, fold_y_test 

def select_data(n_classes, x_tr, x_te, y_tr, y_te):
    if n_classes == 3:
        tri_index_tr = np.where((y_tr==1) | (y_tr==2)|(y_tr==3))[0]
        X_tr_new, y_tr_new = x_tr[tri_index_tr], y_tr[tri_index_tr]
        tri_index_va = np.where((y_te==1) | (y_te==2)|(y_te==3))[0]
        X_te_new, y_te_new = x_te[tri_index_va], y_te[tri_index_va]
        
    if n_classes == 2:
        
        tri_index_tr = np.where((y_tr==1) | (y_tr==2)|(y_tr==3))[0]
        X_tr_new, y_tr_new= x_tr[tri_index_tr], y_tr[tri_index_tr]
        
        tri_index_va = np.where((y_te==1) | (y_te==2)|(y_te==3))[0]
        X_te_new, y_te_new= x_te[tri_index_va], y_te[tri_index_va]
        
        # amusement-> baseline
        index_amuse_tr = np.where((y_tr_new==3))[0]
        y_tr_new[index_amuse_tr] = 1
        index_amuse_te = np.where((y_te_new==3))[0]
        y_te_new[index_amuse_te] = 1
        
     
    return X_tr_new, y_tr_new, X_te_new, y_te_new

def load_case_kemocon(data_path, av_option, class_option):
    if class_option == 2:
        class_option = "2_class"
    else:
        class_option = "3_class"
    path = os.path.join(data_path, class_option)
    X = []
    y = []
    suj = []
    for file in os.listdir(path):
        data = joblib.load(os.path.join(path, file))
        X.append(data["segs"])
        y.append(data[av_option])
        suj.append(np.array([int(file.split('.')[0])]*len(data[av_option])))
        
    X = np.concatenate(X)
    y = np.concatenate(y)
    suj = np.concatenate(suj)
    
    cv = KFold(n_splits = len(np.unique(suj)),  shuffle=False)
    cv_suj = list(cv.split(np.unique(suj)))
    
    return X, y, suj, cv_suj






    
    