# mgcls_dino

[DINO v1 repository](https://github.com/facebookresearch/dino) modified for use with crops from MeerKAT continuum images from the MGCLS survey.

for the paper [Self-Supervised Learning on MeerKAT Wide-Field Continuum Images]()

## Data Availability

Public data can be downloaded from the [MGCLS data release](https://archive-gw-1.kat.ac.za/public/repository/10.48479/7epd-w356/index.html)

## Modifications and additions to original DINO repository

### Modifications 

- main_dino.py -> main_meerkat.py

Minor edits in utils.py to accept single-channel images as input

### Additions

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

Extract features:

```
python eval_knn_train.py --data_path $data_path --dump_features $output_dir --arch $arch --patch_size $patch_size --num_workers 0 --in_chans $in_chans --pretrained_weights $output_dir/$checkpoint_name 
```

Modify config.yaml to point to the desired inputs, labels, and output destination.

```
python evaluator.py --config config.yaml
```

**Attention maps**
```
python visualize_attention.py --image_path prepared_image.npy --image_size 256 256 --arch $arch --output_dir $output_dir --patch_size $patch_size --pretrained_weights $output_dir/$checkpoint_name  --in_chans 1
```


## Pretrained Models

Checkpoints are available for download [here](https://doi.org/10.5281/zenodo.12771941).

| Architecture    | Weight initialization | Pre-training Epochs | Full Checkpoint | Teacher Checkpoint |
| -------- | ------- | -------- | ------- | ------- |
| ViT-S8  | Random | 325 | mgcls_vits8_pretrain_325.pth | mgcls_vits8_pretrain_325_teacher.pth |
| ViT-B16 | DINOv1 | 25 | mgcls_vitb16_pretrain_025.pth | mgcls_vitb16_pretrain_025_teacher.pth |
| ResNet50 | Random | 425 | mgcls_resnet50_pretrain_425.pth | mgcls_resnet50_pretrain_425_teacher.pth |

Use the weights for the teacher network for feature extraction/finetuning

## Documentation

see docstrings


