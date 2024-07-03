import os
import glob
import numpy as np
from transforms import *
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from torchvision.datasets import DatasetFolder

def npy_loader(path):
    sample = np.load(path).astype(np.float32)
    return sample

class RegressionDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        super().__init__()
        self.X = torch.from_numpy(X.astype('float32'))
        self.y = torch.from_numpy(y.astype('float32'))

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index].unsqueeze(0)

class NumpyFolderDataset(DatasetFolder):
    def __init__(self, data_path, indices = None, transform=None, loader = None, fake_3chan = False):
        #super().__init__(data_path, transform=transform, loader = loader)
        if not loader:
            self.loader = npy_loader
        else:
            self.loader = loader 
        if fake_3chan:
            tlist = [transforms.ToTensor()]
            tlist.append(FakeChannels())
            if transform:
                tlist.extend(transform)
            transform = transforms.Compose(tlist)
        self.transform = transform 
        self.data = sorted(glob.glob(os.path.join(data_path,"*.npy")))
        if indices is not None:
            self.data = self.data[indices]
        
    def get_filename(self,index):
        return self.data[index]

    def __getitem__(self, index):
        file = self.data[index]
        x = self.loader(file)

        if self.transform:
            x = self.transform(x) #Image.fromarray(x))  
        return x 
    
    def __len__(self):
        return len(self.data)

class MeerKATDataset(NumpyFolderDataset):
    def __init__(self, data, indices=None, labels=None, transform=None, loader=None, metadata=None):
        super().__init__(data, indices=indices, transform=transform, loader=loader)
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
    def __init__(self, data, labels="source", transform=None, loader=None):
        super().__init__(data, labels=labels, transform=transform, loader = loader)
        
    def label_from_source(self, filename):
        return "COSMOS" if "COSMOS" in filename else "XMMLSS"
    
class MGCLSDataset(MeerKATDataset):
    def __init__(self, data, indices = None, labels=None, transform=None, loader=None, metadata = None, ):
        super().__init__(data, indices=indices, labels=labels, transform=transform, loader = loader, metadata = metadata)
        self.data_product = "basic" if "basic" in data else "enhanced"

# class ReturnIndexDataset(MGCLSDataset):
#     def __getitem__(self, idx):
#         img = super(ReturnIndexDataset, self).__getitem__(idx)
#         return img, idx

