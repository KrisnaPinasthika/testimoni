#!/bin/sh

# python train.py args_train_kitti_eigen.txt
# python train.py --epochs=30 --bs=18 --name=Sparta_S_M --backbone=eff_v2_m --attention_type=weighted --validate-every=100 
python train.py --epochs=25 --bs=28 --name=Exp16_drop035 --validate-every=200 