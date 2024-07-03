
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
import numpy as np
import glob

#sys.path.append('/home/users/l/lastufka/byol')
from byol.models import BYOL
#sys.path.append('/home/users/l/lastufka/fixmatch/main')
#sys.path.append('/home/users/l/lastufka/RadioGalaxyDataset')
from firstgalaxydata import FIRSTGalaxyData
#from fixmatch import evaluation
#from dataloading.datasets import MiraBest_full, MBFRConfident, ReturnIndexDatasetRGZ

import torch
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from torchvision import transforms as pth_transforms
from torchvision import models as torchvision_models

import utils
import vision_transformer as vits
from eval_knn import extract_features
from EvaluationDatasets import ReturnIndexDataset, ReturnIndexDatasetMB, ReturnIndexDatasetF 
from transforms import FakeChannels

def extract_train_feature_pipeline(args):
    """same as in eval_knn.py but only for the train dataset, no labels"""
    if args.in_chans == 1:
        tt = nn.Identity()
    else:
        tt = FakeChannels(additional_channels=args.in_chans)
    # ============ preparing data ... ============
    tlist = [pth_transforms.ToTensor(),tt,pth_transforms.Resize(224, interpolation=3)]
    
    if args.center_crop > 0:
        tlist.append(pth_transforms.CenterCrop(args.center_crop))
    if args.autocontrast == True:
        tlist.append(pth_transforms.RandomAutocontrast(p=1))

    transform = pth_transforms.Compose(tlist)
        
    if "MiraBest" in args.data_path:
        print(f"Evaluating on MiraBest train = {args.train}")
        dataset_train = ReturnIndexDatasetMB(args.data_path, train=args.train, transform=transform)
        print(f"Data loaded with {len(dataset_train)} train imgs.")
    elif "FIRST" in args.data_path:
        print(f"Evaluating on RadioGalaxyDataset = {args.train}")
        tt = 'train' if args.train else 'test'
        dataset_train = ReturnIndexDatasetF(root=args.data_path, selected_split=tt, input_data_list=["galaxy_data_h5.h5"], transform=transform)
    #elif "rgz" in args.data_path:
    #    dataset_train = ReturnIndexDatasetRGZ(args.data_path, train=args.train, transform=transform)
    else:
        dataset_train = ReturnIndexDataset(args.data_path,transform=transform,labels=None, fake_3chan=False, metadata = args.metadata_path, scaling=None)

    sampler = torch.utils.data.DistributedSampler(dataset_train, shuffle=False)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    print(f"Data loaded with {len(dataset_train)} train imgs.")

    # ============ building network ... ============
    if "dinov2" in args.arch:
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg')
    elif "vit" in args.arch:
        if args.patch_size == 14:
            model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
            #print(model)
            
        else:
            model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0, in_chans = args.in_chans)
        print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
    elif "xcit" in args.arch:
        model = torch.hub.load('facebookresearch/xcit:main', args.arch, num_classes=0)
    elif args.arch == "resnet18": #byol
        model = torchvision_models.resnet18().cuda()
        #load state dict
        model_path = args.pretrained_weights
        state_dict = torch.load(model_path, map_location="cuda")['state_dict']
        state_dict = {k.replace("encoder.","") : v for k,v in state_dict.items()}
        state_dict = {k.replace("layers","layer") : v for k,v in state_dict.items()}

        model.load_state_dict(state_dict, strict=True)
        model.eval()
        
        learner = BYOL(
            model,
            image_size = 256,
            hidden_layer = 'avgpool')
        
        _, embedding = learner(data_loader_train, return_embedding = True)
        torch.save(embedding.cpu(), os.path.join(args.dump_features, "trainfeat.pth"))
        return
    elif args.arch in torchvision_models.__dict__.keys():
        model = torchvision_models.__dict__[args.arch](num_classes=0)
        if args.in_chans != 3:
            model.conv1 = nn.Conv2d(args.in_chans, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        if args.arch == "resnet18":
            model.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2), padding=(3, 3), bias=False)

        model.fc = nn.Identity()
        
    else:
        print(f"Architecture {args.arch} non supported")
        sys.exit(1)
    model.cuda()
    utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    model.eval()

    # ============ extract features ... ============
    print("Extracting features for train set...")
    train_features = extract_features(model, data_loader_train, args.use_cuda)
    #train_labels = torch.tensor([s[-1] for s in data_loader_train.samples]).long()
    if utils.get_rank() == 0:
        train_features = nn.functional.normalize(train_features, dim=1, p=2)

   # save features and labels
    if args.dump_features and dist.get_rank() == 0:
        torch.save(train_features.cpu(), os.path.join(args.dump_features, "trainfeat.pth"))
        #torch.save(train_labels.cpu(), os.path.join(args.dump_features, "trainlabels.pth"))

   
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with weighted k-NN on ImageNet')
    parser.add_argument('--batch_size_per_gpu', default=128, type=int, help='Per-GPU batch-size')
    parser.add_argument('--nb_knn', default=[10, 20, 100, 200], nargs='+', type=int,
        help='Number of NN to use. 20 is usually working the best.')
    parser.add_argument('--temperature', default=0.07, type=float,
        help='Temperature used in the voting coefficient')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--use_cuda', default=True, type=utils.bool_flag,
        help="Should we store the features on GPU? We recommend setting this to False if you encounter OOM")
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--world_size', default=1, type=int, help='Patch resolution of the model.')
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--dump_features', default=None,
        help='Path where to save computed features, empty for no saving')
    parser.add_argument('--load_features', default=None, help="""If the features have
        already been computed, where to find them.""")
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--data_path', default='/path/to/imagenet/', type=str)
    #parser.add_argument('--metadata_path', default=None, type=str)
    parser.add_argument('--in_chans', default = 1, type = int, help = 'Length of subset of dataset to use.')
    parser.add_argument('--center_crop', default = 0, type = int, help = 'Length of subset of dataset to use.')
    parser.add_argument('--autocontrast', action='store_true', help = 'Length of subset of dataset to use.')

    parser.add_argument('--train', default=True, type=utils.bool_flag,
        help="Should we store the features on GPU? We recommend setting this to False if you encounter OOM")
    args = parser.parse_args()

    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    if args.load_features:
        train_features = torch.load(os.path.join(args.load_features, "trainfeat.pth"))
        #test_features = torch.load(os.path.join(args.load_features, "testfeat.pth"))
        #train_labels = torch.load(os.path.join(args.load_features, "trainlabels.pth"))
        #test_labels = torch.load(os.path.join(args.load_features, "testlabels.pth"))
    else:
        # need to extract features !
        train_features = extract_train_feature_pipeline(args)

    dist.barrier()
