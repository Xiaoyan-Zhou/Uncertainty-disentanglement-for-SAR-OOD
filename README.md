
# This is a repository for Unified OOD Detection and OOD Generalization for SAR Target Recognition via Uncertainty Disentanglement

## TRAIN our model
stage 1: train backbone and AU branch

```
CUDA_VISIBLE_DEVICES=0 python train_step1_seed.py \
            --base_lr 0.01 --kl_lambda 0.01 --end_epoch 100 \
            --loss_mode_AU ce --batch_size 8 --seed 3407 --save_to "experiments/bs8/seed_3407" 
```
stage 2: train EU branch

```    
CUDA_VISIBLE_DEVICES=0 python train_step2_seed.py \
        --base_lr 0.01 --kl_lambda 0.01 --end_epoch 100 \
        --loss_mode_EU EDL --batch_size 8 --seed 3407 --save_to "experiments/bs8/seed_3407" 
```
