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
    """
    Scale the image via contrast stretching.

    This class implements a transformation to scale images via contrast stretching. It adjusts the intensity range of the image 
    to enhance contrast by stretching pixel values to cover the entire intensity range.

    Parameters:
    plow (float or None): The lower percentile value used as the lower bound for the intensity range.
                          If None, the lower percentile will be calculated from the sample.
    phigh (float or None): The upper percentile value used as the upper bound for the intensity range.
                           If None, the upper percentile will be calculated from the sample.
    center (int or None): The center of the image used for calculating the upper percentile.
                          If None, the entire image will be used.
                          Only applicable when phigh is not None.

    Returns:
    numpy.ndarray: The rescaled image array with values scaled to the range [0, 1].
    """
    def __init__(self, plow= None, phigh = None, center = None):
        self.plow = plow
        self.phigh = phigh
        self.center = center

    def __call__(self, sample):
        """
        Apply contrast stretching transformation to the input sample.

        Parameters:
        sample (numpy.ndarray or torch.Tensor): The input image array.

        Returns:
        numpy.ndarray: The rescaled image array with values scaled to the range [0, 1].
        """
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
    """
    This class implements a transformation to scale images using a power-law transformation.
 
    Parameters:
    C (float or None): The constant value used for normalization.
                       If None, the maximum pixel value in the sample will be used.
    power (float): The power parameter for the power-law transformation.

    Returns:
    numpy.ndarray: The transformed image array with values scaled to the range [-1, 1].
    """
    def __init__(self, C, power):
        self.C = C
        self.power = power

    def __call__(self, sample):
        """
        Apply the power-law transformation to the input sample.

        Parameters:
        sample (numpy.ndarray): The input image array.

        Returns:
        numpy.ndarray: The transformed image array with values scaled to the range [-1, 1].
        """
        sample -= np.nanmin(sample)
        if self.C is None:
            C = torch.max(sample)
        else:
            C = self.C
        print(self.C)
        I_norm = (sample/C)**self.power
        return (I_norm - 0.5)/0.5

class FakeChannels(object):
    """
    Add n channels worth of the same data to the image array to match dimensions of pretrained networks.

    This class implements a transformation to add additional channels of the same data to an image array.
    It is useful for matching the dimensions of pretrained networks that expect a certain number of input channels.

    Parameters:
    additional_channels (int): The number of total channels desired for the output image array. Defaults to 3.

    Returns:
    numpy.ndarray: The image array with additional channels of duplicate data added.
    """
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
