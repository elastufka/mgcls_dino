#import glob
import torch
import numpy as np
from PIL import Image
from torchvision import transforms as tvtransforms
from astropy.visualization import ZScaleInterval
from skimage.exposure import rescale_intensity

class IdentityTransform(object):
    """What it sounds like"""
    def __call__(self, sample):
        return sample

def random_augments(kernel_size = 4):
    #transforms compose of various possibilities
    return tvtransforms.Compose([tvtransforms.RandomHorizontalFlip(p=0.5),
                              tvtransforms.RandomVerticalFlip(p=0.5), 
                              tvtransforms.RandomApply([tvtransforms.GaussianBlur(kernel_size=kernel_size)],p=0.5)])

class PositiveTransform(object):
    """shift image values so no negatives"""
    def __call__(self, sample):
        minv = np.min(sample)
        if minv < 0:
            sample -= minv
        return sample

class Autocontrast(object):
    """shift image values so no negatives"""
    def __call__(self, sample):
        # C H W -> H W C
        #sample = sample.moveaxis(0, -1)
        return tvtransforms.functional.autocontrast(sample)

class ZScaleTransform(object):
    """scale the image via zscale"""
    def __init__(self, zint=None, **kwargs):
        if not zint:
            zint = ZScaleInterval(**kwargs)
        self.zint = zint

    def __call__(self, sample):
        if isinstance(sample, torch.Tensor):
            sample = sample.numpy()
        zscaled_np = self.zint(sample)
        return zscaled_np
    
class ContrastStretchTransform(object):
    """scale the image via contrast stretching"""
    def __init__(self, plow= None, phigh = None, center = None):
        self.plow = plow
        self.phigh = phigh
        self.center = center

    def __call__(self, sample):
        if isinstance(self.plow,int) and isinstance(self.phigh,int):
            self.plow = np.percentile(sample, self.plow)
            if self.center:
                ext = sample.shape[-1]
                if sample.ndim == 3:
                    self.phigh = np.percentile(sample[:(ext//2)-self.center:(ext//2)+self.center,(ext//2)-self.center:(ext//2)+self.center], self.phigh)
                else:
                    self.phigh = np.percentile(sample[(ext//2)-self.center:(ext//2)+self.center,(ext//2)-self.center:(ext//2)+self.center], self.phigh)
            else:
                self.phigh = np.percentile(sample, self.phigh)
        if isinstance(sample, torch.Tensor):
            sample = sample.numpy()
        rescaled_np = rescale_intensity(sample, in_range=(self.plow, self.phigh), out_range= (0,1))
        return rescaled_np
    
class PowerlawTransform(object):
    """scale the image via Mariia's method"""
    def __init__(self, C, power):
        self.C = C
        self.power = power

    def __call__(self, sample):
        sample -= np.nanmin(sample)
        if self.C is None:
            C = torch.max(sample)
        else:
            C = self.C
        print(self.C)
        I_norm = (sample/C)**self.power
        return (I_norm - 0.5)/0.5

class FakeChannels(object):
    """Add n channels worth of the same data to the image array to match dimensions of pretrained networks"""
    def __init__(self, additional_channels =3):
        self.additional_channels = additional_channels
    def __call__(self, sample):
        return sample.repeat(self.additional_channels, 1, 1)

class NaNtoZero(object):
    """transform NaNs to 0"""
    def __call__(self, sample):
        new_sample = np.nan_to_num(sample)
        return new_sample

class Rescale(object):
    """rescale rgz images from 0,255 to 0,1"""
    def __call__(self,sample):
        return (np.array(sample)/255).astype(np.float32)

def mblabels(label):
    """transform labels  0,1,2,3,4 into 0 for FRI, 5,6,7 into 1 for FRII, 8,9 into 2 for hybrid"""
    if label in range(5):
        return 0
    if label in [5,6,7]:
        return 1
    if label in [8,9]:
        return 2
