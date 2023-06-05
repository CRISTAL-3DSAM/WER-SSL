# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 11:47:03 2022

@author: WU
"""



from torch.utils.data import Dataset

class PresageDataset(Dataset):

    def __init__(self, data, labels=None, transform=None):

   
        self.transform = transform
            
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Use only 3 signal modalities
        sample = self.data[idx,:, :3]

        if self.transform:
            sample = self.transform(sample)

        return sample   
    
class SupervisedDatasets(Dataset):

    def __init__(self, data, labels, transform=None):

   
        self.transform = transform
            
        self.data = data
        self.label = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        sample = {'eda': self.data[idx,:, 0], 'bvp': self.data[idx,:, 1], 
                  'temp': self.data[idx,:, 2], 'label': self.label[idx]}

        if self.transform:
            sample = self.transform(sample)

        return sample   
    
    