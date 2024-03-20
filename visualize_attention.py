# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import argparse
import cv2
import random
import colorsys
import requests
from io import BytesIO

import skimage.io
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
import numpy as np
from PIL import Image
#sys.path.append('/home/glados/unix-Documents/AstroSignals/feuerzeug')
sys.path.append('/home/users/l/lastufka/dinov2')
from dinov2.models.vision_transformer import vit_small, vit_large, vit_giant2
from feuerzeug.transforms import FakeChannels
#from map_utils import mapdata_from_fits

import utils
import vision_transformer as vits
import wandb


def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
    return image


def random_colors(N, bright=True):
    """
    Generate random colors.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def display_instances(image, mask, fname="test", figsize=(5, 5), blur=False, contour=True, alpha=0.5):
    fig = plt.figure(figsize=figsize, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax = plt.gca()

    N = 1
    mask = mask[None, :, :]
    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    margin = 0
    ax.set_ylim(height + margin, -margin)
    ax.set_xlim(-margin, width + margin)
    ax.axis('off')
    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]
        _mask = mask[i]
        if blur:
            _mask = cv2.blur(_mask,(10,10))
        # Mask
        masked_image = apply_mask(masked_image, _mask, color, alpha)
        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        if contour:
            padded_mask = np.zeros((_mask.shape[0] + 2, _mask.shape[1] + 2))
            padded_mask[1:-1, 1:-1] = _mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8), aspect='auto')
    fig.savefig(fname)
    print(f"{fname} saved.")
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualize Self-Attention maps')
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base', 'dinov2'], help='Architecture (support only ViT atm).')
    parser.add_argument('--patch_size', default=8, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='', type=str,
        help="Path to pretrained weights to load.")
    parser.add_argument('--pretrain_epochs', default=0, type=int, help='Number of epochs on which backbone was trained')
    parser.add_argument('--pretrain_data', default="MGCLS subset", type=str, help='date on which pre-trained backbone was trained')
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument("--image_path", default=None, type=str, help="Path of the image to load.")
    parser.add_argument("--prefix", default="", type=str, help="Path of the image to load.")
    parser.add_argument("--image_size", default=(480, 480), type=int, nargs="+", help="Resize image.")
    parser.add_argument('--output_dir', default='.', help='Path where to save visualizations.')
    parser.add_argument("--threshold", type=float, default=None, help="""We visualize masks
        obtained by thresholding the self-attention maps to keep xx% of the mass.""")
    parser.add_argument("--index", type=int, default=None, help="""Index of the numpy array to use as image""")
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # build model
    if "dinov2" in args.arch:
        #model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg')
        model = vit_giant2(
            patch_size=14,
            img_size=526,
            init_values=1.0,
            #ffn_layer="mlp",
            block_chunks=0)
    #model.load_state_dict(torch.load('dinov2_vitl14_pretrain.pth'))

    else:
        model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.to(device)
    if os.path.isfile(args.pretrained_weights):
        state_dict = torch.load(args.pretrained_weights, map_location="cpu")
        if args.checkpoint_key is not None and args.checkpoint_key in state_dict:
            print(f"Take key {args.checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[args.checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(args.pretrained_weights, msg))
    else:
        print("Please use the `--pretrained_weights` argument to indicate the path of the checkpoint to evaluate.")
        url = None
        if args.arch == "vit_small" and args.patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif args.arch == "vit_small" and args.patch_size == 8:
            url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"  # model used for visualizations in our paper
        elif args.arch == "vit_base" and args.patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif args.arch == "vit_base" and args.patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        if url is not None:
            print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
            state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
            model.load_state_dict(state_dict, strict=True)
        else:
            print("There is no reference weights available for this model => We use random weights.")

    # open image
    if args.image_path is None:
        # user has not specified any image - we use our own image
        print("Please use the `--image_path` argument to indicate the path of the image you wish to visualize.")
        print("Since no image path have been provided, we take the first image in our paper.")
        response = requests.get("https://dl.fbaipublicfiles.com/dino/img.png")
        img = Image.open(BytesIO(response.content))
        img = img.convert('RGB')
    elif os.path.isfile(args.image_path):
        if args.image_path.endswith(".npy"):
            imarr = np.load(args.image_path).astype(np.float32)
            #if not args.index:
            #    idx = np.random.randint(len(imarr))
            #else: 
            #    idx = args.index
            #print(f"image index: {idx}")
            #img = imarr[idx]
            img = Image.fromarray(imarr)
        else: #FITS file
            _, img = mapdata_from_fits(args.image_path)
            img = img.astype(np.float32)
        #are there NaN?
        if np.isnan(np.mean(img)):
            print("NaN found!")
            #print(f"shape: {img.shape}")
            img = np.nan_to_num(img)
        norm_mean = np.nanmean(img)
        norm_std = np.nanstd(img)
        #print(norm_mean, np.mean(img))
            #print(img.shape)
        #img = torch.from_numpy(npimg) 
        # with open(args.image_path, 'rb') as f:
        #     img = Image.open(f)
        #     img = img.convert('RGB')
    else:
        print(f"Provided image path {args.image_path} is non valid.")
        sys.exit(1)
    transform = pth_transforms.Compose([
        pth_transforms.ToTensor(),
        FakeChannels(),
        pth_transforms.RandomAutocontrast(p=1)
        #pth_transforms.Resize(args.image_size),
        #pth_transforms.Normalize((norm_mean,1e-6,1e-6),(norm_std,1e-6,1e-6)), #(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    ])
    img = transform(img)

    # make the image divisible by the patch size
    w, h = img.shape[1] - img.shape[1] % args.patch_size, img.shape[2] - img.shape[2] % args.patch_size
    img = img[:, :w, :h].unsqueeze(0)

    w_featmap = img.shape[-2] // args.patch_size
    h_featmap = img.shape[-1] // args.patch_size

    attentions = model.get_last_self_attention(img.to(device))

    nh = attentions.shape[1] # number of head

    # we keep only the output patch attention
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

    if args.threshold is not None:
        # we keep only a certain percentage of the mass
        val, idx = torch.sort(attentions)
        val /= torch.sum(val, dim=1, keepdim=True)
        cumval = torch.cumsum(val, dim=1)
        th_attn = cumval > (1 - args.threshold)
        idx2 = torch.argsort(idx)
        for head in range(nh):
            th_attn[head] = th_attn[head][idx2[head]]
        th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
        # interpolate
        th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=args.patch_size, mode="nearest")[0].cpu().numpy()

    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=args.patch_size, mode="nearest")[0].cpu().numpy()

    
    #save attentions heatmaps
    os.makedirs(args.output_dir, exist_ok=True)
    torchvision.utils.save_image(torchvision.utils.make_grid(img, normalize=True, scale_each=True), os.path.join(args.output_dir, f"{args.prefix}img.png"))
    for j in range(nh):
        fname = os.path.join(args.output_dir, f"{args.prefix}attn-head{str(j)}.png")
        plt.imsave(fname=fname, arr=attentions[j], format='png')
        print(f"{fname} saved.")
        
#     wandb.init(
#         # set the wandb project where this run will be logged
#         project="visualize_attention",
#         group= f"{args.arch}_{args.pretrain_epochs}_{args.pretrain_data}",
#         # track hyperparameters and run metadata
#         config={
#         #"learning_rate": hps['lr'],
#         "pre-trained backbone": args.arch,
#         "patch size": args.patch_size,
#         "pre-training epochs": args.pretrain_epochs,
#         "pre-training dataset": args.pretrain_data,
#         "image" : args.image_path,
#         }
#     )   
    
#     fig, ax = plt.subplots(2,4, figsize = (10,6))
#     imarr = np.array(img)[0,0,:,:]
#     ax[1,2].imshow(imarr, origin='lower')
#     ax[1,2].set_title("Img")
#     ax[1,2].axis("off")
    
#     ax[1,3].imshow(np.log10(imarr), origin='lower')
#     ax[1,3].set_title("log10 img")
#     ax[1,3].axis("off")


#     for i in range(nh):
#         im = attentions[i]
#         ax[i//4,i%4].imshow(im,origin='lower')
#         head_title = f"attn-head {i}"
#         ax[i//4,i%4].set_title(head_title, fontsize = 11)
#         ax[i//4,i%4].axis('off')
    
#     wandb.log({"attention-maps":fig})
#     wandb.finish()

    if args.threshold is not None:
        image = skimage.io.imread(os.path.join(args.output_dir, f"{args.prefix}img.png"))
        for j in range(nh):
            display_instances(image, th_attn[j], fname=os.path.join(args.output_dir, "mask_th" + str(args.threshold) + "_head" + str(j) +".png"), blur=False)
