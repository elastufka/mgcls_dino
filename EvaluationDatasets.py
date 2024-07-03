import sys
import torch
sys.path.append('/path/to/fixmatch/main')
from dataloading.datasets import MiraBest_full, MBFRConfident, ReturnIndexDatasetRGZ
sys.path.append('/path/to/RadioGalaxyDataset')
from firstgalaxydata import FIRSTGalaxyData
from MeerKATDataset import MGCLSDataset

class RegressionDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        super().__init__()
        self.X = torch.from_numpy(X.astype('float32'))
        self.y = torch.from_numpy(y.astype('float32'))

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index].unsqueeze(0)

class ReturnIndexDataset(MGCLSDataset):
    def __getitem__(self, idx):
        img = super(ReturnIndexDataset, self).__getitem__(idx)
        return img, idx

class ReturnIndexDatasetMB(MBFRConfident):
    def __getitem__(self, idx):
        img, _ = super(ReturnIndexDatasetMB, self).__getitem__(idx)
        return img, idx

class ReturnIndexDatasetF(FIRSTGalaxyData):
   def __getitem__(self, idx):
       img, _ = super(ReturnIndexDatasetF, self).__getitem__(idx)
       return img, idx
   
class RegressionDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        super().__init__()
        self.X = torch.from_numpy(X.astype('float32'))
        self.y = torch.from_numpy(y.astype('float32'))

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index].unsqueeze(0)