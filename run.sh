#!/bin/sh

######################
# Example run scripts
######################
seed=1

# SFT on the VS10 dataset
standard_options="--seed ${seed} --dataset vgg --num_epochs 20 --batch_size 24 --optim AdamW --n_classes 10 --lr 1e-3 --delay_stop 6 --n_layers_pre_fusion 1 --n_layers_post_fusion 11 --custom_ks"
python main.py $standard_options --exp_name stride64 --model sft --pool_type max --warmup_epochs 5 --mixup any --alpha_type permodality --ks_modality 38,64,38 --debug

# SFT on the VS100 dataset
standard_options="--seed ${seed} --dataset vgg --num_epochs 20 --batch_size 24 --optim AdamW --n_classes 100 --lr 1e-3 --delay_stop 6 --n_layers_pre_fusion 1 --n_layers_post_fusion 11 --custom_ks"
python main.py $standard_options --exp_name stride64 --model sft --pool_type max --warmup_epochs 5 --mixup any --alpha_type permodality --ks_modality 38,64,38 --debug

# SFT on MOSEI
standard_options="--seed ${seed} --dataset mosei_senti --num_epochs 20 --batch_size 24 --optim AdamW --n_classes 7 --lr 1e-4 --delay_stop 6 --n_layers_pre_fusion 1 --n_layers_post_fusion 11 --custom_ks"
python main.py $standard_options --exp_name stride64 --model sft --pool_type max --warmup_epochs 5 --mixup any --alpha_type permodality --ks_modality 64,64,50 --debug

