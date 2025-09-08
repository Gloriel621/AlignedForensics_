#DATASET=path to dataset
# DATASET="ENTER DATASET PATH"
EXP_NAME="staypos-reimplement-cocolsun-180k"
#LDM_DS_NAME=directory of checkpoints

# Train on 1000 real and 1000 fake in the sync mode (use half the usual batch size as the code picks a fake image for every real image)
#CUDA_VISIBLE_DEVICES=2 python train.py --name=$LDM_DS_NAME --arch res50nodown --cropSize 96  --norm_type resnet --resize_size 256 --resize_ratio 0.75 --blur_sig 0.0,3.0 --cmp_method cv2,pil --cmp_qual 30,100 --resize_prob 0.2 --jitter_prob  0.8 --colordist_prob 0.2 --cutout_prob 0.2 --noise_prob 0.2 --blur_prob 0.5 --cmp_prob 0.5 --rot90_prob 1.0 --dataroot $DATASET --batch_size 32 --earlystop_epoch 10 --use_inversions --seed 17 --batched_syncing --data_cap=1000

# Train on whole dataset without sync mode
#CUDA_VISIBLE_DEVICES=2 python train.py --name=$LDM_DS_NAME --arch res50nodown --cropSize 96  --norm_type resnet --resize_size 256 --resize_ratio 0.75 --blur_sig 0.0,3.0 --cmp_method cv2,pil --cmp_qual 30,100 --resize_prob 0.2 --jitter_prob  0.8 --colordist_prob 0.2 --cutout_prob 0.2 --noise_prob 0.2 --blur_prob 0.5 --cmp_prob 0.5 --rot90_prob 1.0 --dataroot $DATASET --batch_size 128 --earlystop_epoch 10 --use_inversions --seed 17 

REAL_DIR="/media/NAS/USERS/gloriel621/DF_dataset/af_trainset_180k/real/"
FAKE_DIR="/media/NAS/USERS/gloriel621/DF_dataset/af_trainset_180k/fake/"

# ICML experiments (Stay-Positive with batch level alignment)
CUDA_VISIBLE_DEVICES=1 python train.py \
    --name=$EXP_NAME \
    --arch res50nodown \
    --data_cap=200000 \
    --real_dir $REAL_DIR \
    --fake_dir $FAKE_DIR \
    --val_split 0.1 \
    --cropSize 96 \
    --norm_type resnet \
    --resize_size 256 \
    --resize_ratio 0.75 \
    --blur_sig 0.0,3.0 \
    --cmp_method cv2,pil \
    --cmp_qual 30,100 \
    --resize_prob 0.2 \
    --jitter_prob  0.8 \
    --colordist_prob 0.2 \
    --cutout_prob 0.2 \
    --noise_prob 0.2 \
    --blur_prob 0.5 \
    --cmp_prob 0.5 \
    --rot90_prob 1.0 \
    --batch_size 32 \
    --earlystop_epoch 5 \
    --seed 14 \
    --stay_positive "clamp"\
    --fix_backbone \
    --final_dropout 0.0 \
    --use_inversions \
    --batched_syncing \
    --ckpt /media/NAS/USERS/gloriel621/DF_methods/AlignedForensics/test_code/weights/ours-sync/weights.pth \

