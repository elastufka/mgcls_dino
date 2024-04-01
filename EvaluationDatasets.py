
import sys
sys.path.append('/home/users/l/lastufka/fixmatch/main')
from dataloading.datasets import MiraBest_full, MBFRConfident, ReturnIndexDatasetRGZ
sys.path.append('/home/users/l/lastufka/RadioGalaxyDataset')
from firstgalaxydata import FIRSTGalaxyData
from MeerKATDataset import MGCLSDataset

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