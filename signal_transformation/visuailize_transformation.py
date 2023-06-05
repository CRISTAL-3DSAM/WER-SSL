# -*- coding: utf-8 -*-

import numpy as np

import matplotlib.pyplot as plt

import signal_transformation_task as stt

def transform(sample):
    
    trans_arr = np.zeros((6, sample.shape[-1]))
    noised = stt.add_noise_with_SNR(sample, noise_amount =15)
    permuted = stt.permute(sample, pieces = 10)
    time_wraped = stt.time_warp_v3(sample, sampling_freq = 4,
                                                  pieces = 9, stretch_factor =1.5,
                                                  squeeze_factor = 1/1.5)
    crop_resized = stt.CropResize(sample, nPerm =4)
    MagWarpped = stt.MagWarp(sample, sigma = 0.2)
    trans_arr =  np.array([sample, noised, MagWarpped, permuted, time_wraped, crop_resized])
    
    return trans_arr

# a EDA sample of 20 minutes
X = np.load(r"sample.npy")
trans = transform(X)
plt.style.use('default')    

heights = [105]*2
widths = [228*9]
plt.rcParams['pdf.fonttype'] = 42
colors=[
    '#1f77b4', '#aec7e8', 
    '#ff7f0e', '#ffbb78', 
    '#2ca02c', '#d62728',
    '#c5b0d5', '#e377c2'
    ]

trans_list = ['Original', 'Noise addition', 'Magnitude-warping', 'Permutation', 'Time-wrapping', 'Cropping']
trans_id = 0

# create a figure
f = plt.figure()
f.set_figheight(3)
f.set_figwidth(16)

import matplotlib.gridspec as gridspec
spec = gridspec.GridSpec(ncols=3, nrows= 2,
                         width_ratios=[1, 1, 1], wspace=0.3,
                          hspace=1.1)
axarr = []
for i in range(6):
    axarr.append(f.add_subplot(spec[i]))

import matplotlib.ticker as ticker
labels = ["0", "5", "10", "15", "20"]

for i in range(6):
            axarr[i].plot(np.arange(trans.shape[-1]), trans[trans_id], c = colors[trans_id])
            axarr[i].tick_params(labelsize=11)
            axarr[i].set_xlim([0, len(X)])
            axarr[i].xaxis.set_major_locator(ticker.FixedLocator(np.arange(0,len(X)+len(X)//4,len(X)//4)))
            axarr[i].xaxis.set_major_formatter(ticker.FixedFormatter(labels))
            axarr[i].set_ylabel(r"$\mu$S", fontsize =12)
            axarr[i].grid()
            axarr[i].set_title(trans_list[trans_id], size=14)
            axarr[i].set_xlabel("Time (min)", fontsize =12)
            trans_id += 1

plt.tight_layout()

