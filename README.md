# mgcls_dino

DINO v1 repository (https://github.com/facebookresearch/dino) modified for use with crops from MeerKAT continuum images from the MGCLS survey.

for the paper [Beyond Galaxy Zoo: Self-Supervised Learning on MeerKAT Wide-Field Continuum Images]()

## Data Availability

Public data can be downloaded from the [MGCLS data release](https://archive-gw-1.kat.ac.za/public/repository/10.48479/7epd-w356/index.html)

## Modifications and additions to original DINO repository

### Modifications 

- main_dino.py -> main_meerkat.py

Minor edits in utils.py to accept single-channel images as input

### Additionals

**Data preparation**
- mgcls_data_prep.py

**Training with custom dataset**
- MeerKATDataset.py
- transforms.py

**Feature extraction**
- eval_knn_train.py

**Evaluation based on extracted features**
- evaluator.py
- eval.yaml
- evaluation_models.py
- EvaluationDatasets.py

## Requirements

To perform the evaluation as in the paper, it is required to install the following repositories:
- [MiraBest](https://github.com/inigoval/fixmatch)
- [FIRST](https://github.com/floriangriese/RadioGalaxyDataset)

To generate COCO-format labels for compact sources, the following repository is required:
- [pyBDSF_to_COCO](https://github.com/elastufka/pyBDSF_to_COCO)

## Installation

Clone or fork the repository.

## Usage

**Data preparation**

```
python mgcls_data_prep.py --data_path /path/to/FTIS/files
```

**Training**

```
python main_meerkat.py --data_path $train --output_dir $output_dir --arch $arch --patch_size $patch_size --epochs $epochs --saveckp_freq $savefrq --num_workers 0 --batch_size_per_gpu $batch_size_per_gpu --use_fp16 false --momentum_teacher 0.996 --augmentations rotate powerlaw --weight_decay $weight_decay --weight_decay_end $weight_decay_end --lr $lr --in_chans 1 --project $project --checkpoint_name $checkpoint_name
```

if using more than one gpu and SLURM:

```
srun python -m torch.distributed.launch --nproc_per_node=4 main_meerkat.py --data_path $train --output_dir $output_dir --arch $arch --patch_size $patch_size --epochs $epochs --saveckp_freq $savefrq --num_workers 0 --batch_size_per_gpu $batch_size_per_gpu --use_fp16 false --momentum_teacher 0.996 --augmentations rotate powerlaw --weight_decay $weight_decay --weight_decay_end $weight_decay_end --lr $lr --in_chans 1 --project $project --checkpoint_name $checkpoint_name
```

**Evaluation**

Generate labels via pyBDSF_to_COCO and crop_catalog_aggs(). Otherwise modify evaluator.py to accept the names of custom labels.

Modify config.yaml to point to the desired inputs, labels, and output destination.

```
python evaluator.py --config config.yaml
```

## Pretrained Models

| Architecture    | Weight initialization | Pre-training Epochs | Checkpoint name |
| -------- | ------- | -------- | ------- |
| ViT-S8  | Random | 325 | mgcls_vits8_pretrain_325.pth |
| ViT-B16 | DINOv1 | 25 | mgcls_vitb16_pretrain_025.pth |
| ResNet50 | Random | 425 | mgcls_resnet50_pretrain_425.pth |

## Documentation

see docstrings

## Acknowledgements

Credits to the original authors and contributors of Facebook's DINO framework.
Acknowledgements to any other sources or individuals who contributed to your modifications.


