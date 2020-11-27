# Predicting Short-Term Next-Active-Object Through Visual Attention and Hand Position

This repository is code of [NAO]().

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Download data annotation

- [ADL Annotation](https://mega.nz/folder/7EoWRB4L#cJvx_sPShW8YXFBNfK9TsQ)
- [EPIC Annotation](https://mega.nz/folder/zRwExbIT#UDD0OztUoxc95yWdOgad4g)

## Training and Evaluation

To train/evaluate the model(s) on EPIC dataset, run this command:

```train
python train.py --dataset EPIC --exp_name epic/unet_resnet_hand_att --lr 0.0000001 --bs 12
python test.py --dataset EPIC --exp_name epic/unet_resnet_hand_att  --bs 12
```

To train/evaluate the model(s) on ADL dataset, run this command:

```train
python train.py --dataset ADL --exp_name adl/unet_resnet_hand_att --lr 0.0000002
 --bs 12
python test.py --dataset ADL --exp_name adl/unet_resnet_hand_att  --bs 12
```

## Pre-trained Models

You can download pretrained models here:

- [pre-trained model](https://mega.nz/file/fEwkVbYQ#8FAaGcSlTV3QNgboNXRW4JMzv8IjRx8bY6MMh_BB6f4) trained on EPIC-Kitchen. 

- [pre-trained model](https://mega.nz/file/GNpWnZwa#SZflnw7e4gEWHvtznMbSjViWsebu59xOMa4CN0g8Zpg) trained on ADL. 

