#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=4 
export DATA_ROOT=/data/zhicai/dataset/ZSL_data
export DATASET=CUB
export NCLASS_ALL=200
export ATTSIZE=312
export SYN_NUM=400
export SYN_NUM2=1000
export RESSIZE=2048
export BATCH_SIZE=64
export IMAGE_EMBEDDING=res101
export CLASS_EMBEDDING=att
export NEPOCH=300
export GAMMAD=10
export GAMMAG=10
export gammaD_un=1000
export gammaG_un=10
export GAMMAD_ATT=10
export GAMMAG_ATT=0.1
export LAMBDA1=10
export CRITIC_ITER=5 
export LR=0.0002
export CLASSIFIER_LR=0.001
export MSE_WEIGHT=1
export RADIUS=1
export MANUALSEED=3483
export NZ=312
export BETA=10 # L_R weight
seed=(4115)
beta=(10)
for six in $(seq 1 1 ${#seed[@]}); do
    for six2 in $(seq 1 1 ${#beta[@]}); do
        export MANUALSEED=${seed[((six-1))]}
            export BETA=${beta[((six2-1))]}
            python -u train.py \
                --cuda \
                --RCritic \
                --L2_norm \
                --transductive \
                --encoded_noise \
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
            # 1>/dev/null 2>&1 &\
    done
done
