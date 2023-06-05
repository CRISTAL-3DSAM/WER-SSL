# -*- coding: utf-8 -*-
"""
Codes adapted from https://code.engineering.queensu.ca/pritam/SSL-ECG/-/blob/master/implementation/signal_transformation_task.py
and https://github.com/terryum/Data-Augmentation-For-Wearable-Sensor-Data

"""
import cv2
import math
import random
import numpy as np
from scipy import signal
from scipy.interpolate import CubicSpline

def add_noise_with_SNR(signal, noise_amount):
    """ 
    adding noise
    created using: https://stackoverflow.com/a/53688043/10700812 
    """
    
    target_snr_db = noise_amount #20
    x_watts = signal ** 2                       # Calculate signal power and convert to dB 
    sig_avg_watts = np.mean(x_watts)
    sig_avg_db = 10 * np.log10(sig_avg_watts)   # Calculate noise then convert to watts
    noise_avg_db = sig_avg_db - target_snr_db
    noise_avg_watts = 10 ** (noise_avg_db / 10)
    mean_noise = 0
    noise_volts = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), len(x_watts))     # Generate an sample of white noise
    noised_signal = signal + noise_volts        # noise added signal

    return noised_signal 

def GenerateRandomCurves(X, sigma=0.2, knot=4):
    xx = np.arange(0,X.shape[0], (X.shape[0]-1)/(knot+1))
    yy = np.random.normal(loc=1.0, scale=sigma, size=(knot+2,))
    x_range = np.arange(X.shape[0])
    cs_x = CubicSpline(xx, yy)

    return cs_x(x_range)
    
def MagWarp(X, sigma = 0.2):
    return X * GenerateRandomCurves(X, sigma)

def permute(signal, pieces):
    """ 
    signal: numpy array (batch x window)
    pieces: number of segments along time    
    """
    split_signal = np.array_split(signal, pieces)
    order = list(range(0,pieces))
    np.random.shuffle(order)
    
    return np.hstack([split_signal[s] for s in order])


def time_warp_v3(signal, sampling_freq, pieces, stretch_factor, squeeze_factor):
    """ 
    signal: numpy array (batch x window)
    sampling freq
    pieces: number of segments along time
    stretch factor
    squeeze factor
    """
    
    total_time = np.shape(signal)[0]//sampling_freq
    segment_time = total_time/pieces
    sequence = list(range(0,pieces))
    stretch = np.random.choice(sequence, math.ceil(len(sequence)/2), replace = False)
    squeeze = list(set(sequence).difference(set(stretch)))
    initialize = True
    for i in sequence:
        orig_signal = signal[int(i*np.floor(segment_time*sampling_freq)):int((i+1)*np.floor(segment_time*sampling_freq))]
        orig_signal = orig_signal.reshape(np.shape(orig_signal)[0],1)
        stretch_shape = int(np.ceil(np.shape(orig_signal)[0]*stretch_factor))
        squeeze_shape1 = int((np.shape(signal)[0] - (len(stretch))*stretch_shape)/len(squeeze))
        squeeze_shape2 = np.shape(signal)[0] - (len(stretch))*stretch_shape  - (len(squeeze)-1)*squeeze_shape1
        if i in stretch:
            
            new_signal = cv2.resize(orig_signal, (1, stretch_shape), interpolation=cv2.INTER_LINEAR)
            if initialize == True:
                time_warped = new_signal
                initialize = False
            else:
                time_warped = np.vstack((time_warped, new_signal))
        elif i in squeeze:
            if i != squeeze[-1]:
                new_signal = cv2.resize(orig_signal, (1, squeeze_shape1), interpolation=cv2.INTER_LINEAR)
                if initialize == True:
                    time_warped = new_signal
                    initialize = False
                else:
                    time_warped = np.vstack((time_warped, new_signal))
            else:
                new_signal = cv2.resize(orig_signal, (1, squeeze_shape2), interpolation=cv2.INTER_LINEAR)
                if initialize == True:
                    time_warped = new_signal
                    initialize = False
                else:
                    time_warped = np.vstack((time_warped, new_signal))
    return time_warped.flatten()

def CropResize(X, nPerm =4):
    n_pieces       = int(np.ceil(X.shape[0]/(X.shape[0]//nPerm)))
    piece_length = int(X.shape[0]//n_pieces)
    
    pieces = [X[i*piece_length: (i+1)*piece_length] for i in range(n_pieces)]
    
    idx_selected = random.choice(list(range(n_pieces)))
    X_new = signal.resample(pieces[idx_selected], X.shape[0])
    return X_new