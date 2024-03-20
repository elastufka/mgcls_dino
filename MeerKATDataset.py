import os
import glob
import numpy as np
from transforms import *
import pandas as pd

class MeerKATDataset(NumpyFolderDataset):
    def __init__(self, data, indices=None, labels=None, transform=None, loader=None, metadata=None):
        super().__init__(data, indices=indices, labels=labels, transform=transform, loader=loader, metadata=metadata)
        #self.transform = transform
        self.labels_from = labels
        if metadata is not None:
            self.metadata = pd.read_csv(metadata)
        
    def __getitem__(self, index):
        file = self.data[index]
        x = self.loader(file)

        if self.transform:
            x = self.transform(x) # Image.fromarray(x)) 
        if self.labels_from:
            y = self.label_from_metadata(file, index)    
            return x, y
        return x
    
    def label_from_metadata(self, file, index):
        """get the label from the metadata dataframe"""
        filename = file[file.rfind('/')+1:]
        labels = self.metadata.where(self.metadata.filename_enhanced == filename).dropna(how='all')[self.labels_from]
        try:
            label = labels.iloc[0]
        except Exception as e:
            print(filename, labels)
        return label

class MIGHTEEDataset(MeerKATDataset):
    def __init__(self, data, labels="source", transform=None, loader=None, fake_3chan = True, scaling = 'contrast'):
        super().__init__(data, labels=labels, transform=transform, loader = loader, fake_3chan = fake_3chan, scaling = scaling)
        
    def label_from_source(self, filename):
        return "COSMOS" if "COSMOS" in filename else "XMMLSS"
    
class MGCLSDataset(MeerKATDataset):
    def __init__(self, data, indices = None, labels=None, transform=None, loader=None, fake_3chan = True, metadata = None, scaling = 'contrast'):
        super().__init__(data, indices=indices, labels=labels, transform=transform, loader = loader, fake_3chan = fake_3chan, metadata = metadata, scaling = scaling)
        self.data_product = "basic" if "basic" in data else "enhanced"
        
class MGCLSPairedDataset(PairedDataset):
    def __init__(self, data_path0, data_path1, transform0=None, transform1=None, loader=None):
        super().__init__(data_path0, data_path1, transform0=transform0, transform1=transform1, loader = loader)
  
    def __getitem__(self, index):
        '''Assume DINO transforms, so combine global/local crops before output'''
        file0, file1 = self.data[index]
        x0 = self.loader(file0)
        x1 = self.loader(file1)

        if self.transform0:
            x0 = self.transform0(Image.fromarray(x0))
        if self.transform1:
            x1 = self.transform1(Image.fromarray(x1))  
            
        # global crops together
        g0 = x0['global']
        g1 = x1['global'][0] #only take the first global crop of the enhanced image. it will go to the student network
        g0.extend(g1)
        #np.random.shuffle(g0)
        # local crops together
        l0 = x0['local']
        l1 = x1['local']
        l0.extend(l1)
        np.random.shuffle(l0)
        #put them all in one list
        g0.extend(l0)

        return g0

class ReturnIndexDataset(MGCLSDataset):
    def __getitem__(self, idx):
        img = super(ReturnIndexDataset, self).__getitem__(idx)
        return img, idx

