#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=2
export DATA_ROOT=/home/zhicai/data
export DATASET=SUN
export NCLASS_ALL=717
export ATTSIZE=102
export SYN_NUM=400
export SYN_NUM2=1000
export RESSIZE=2048
export BATCH_SIZE=64
export IMAGE_EMBEDDING=res101
export CLASS_EMBEDDING=att
export NEPOCH=600
export GAMMAD=100
export GAMMAG=1
export gammaD_un=100
export gammaG_un=1
export GAMMAD_ATT=1
export GAMMAG_ATT=0.01
export LAMBDA1=10
export CRITIC_ITER=5
export LR=0.0002
export CLASSIFIER_LR=0.001
export MSE_WEIGHT=1
export RADIUS=1
export MANUALSEED=3115
export NZ=102
export BETA=10 # L_R weight
export ATT_POWER=1.0
# --with_norm_weight\
seed=(4115 3254) 
r=(1)
ar=(1)
att_power=(1)
for six in $(seq 1 1 ${#seed[@]}); do
    for six2 in $(seq 1 1 ${#r[@]}); do
    export MANUALSEED=${seed[((six-1))]}
    export RADIUS=${r[((six2-1))]}
    python -u vaeganV1.py \
        --cuda \
        --perb \
        --RCritic \
        --L2_norm \
        --encoded_noise \
        --transductive \
        --manualSeed $MANUALSEED \
        --nclass_all $NCLASS_ALL \
        --dataroot $DATA_ROOT \
        --beta $BETA \
        --dataset $DATASET \
        --batch_size $BATCH_SIZE \
        --attSize $ATTSIZE \
        --resSize $RESSIZE \
        --image_embedding $IMAGE_EMBEDDING \
        --class_embedding $CLASS_EMBEDDING \
        --syn_num $SYN_NUM \
        --syn_num2 $SYN_NUM2 \
        --nepoch $NEPOCH \
        --gammaD $GAMMAD \
        --gammaG $GAMMAG \
        --gammaD_un $gammaD_un \
        --gammaG_un $gammaG_un \
        --gammaD_att $GAMMAD_ATT \
        --gammaG_att $GAMMAG_ATT \
        --lambda1 $LAMBDA1 \
        --critic_iter $CRITIC_ITER \
        --lr $LR \
        --classifier_lr $CLASSIFIER_LR \
        --mse_weight $MSE_WEIGHT\
        --radius $RADIUS\
    1>/dev/null 2>&1 &\
    done
done

