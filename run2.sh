#!/bin/sh

python -u -c 'import torch; print(torch.__version__)'

CODE_PATH=codes
DATA_PATH=data
SAVE_PATH=models
LOG_PATH=logs

#The first four parameters must be provided
#MODE=$1
MODEL=$1
DATASET=$2
GPU_DEVICE=$3
SAVE_ID=$4
LOG_ID=$5

FULL_DATA_PATH=$DATASET
SAVE=$SAVE_PATH/"$MODEL"_"$DATASET"_"$SAVE_ID"
LOG=$LOG_PATH/"$MODEL"_"$DATASET"_"$LOG_ID"

#Only used in training
BATCH_SIZE=$6
NEGATIVE_SAMPLE_SIZE=$7
HIDDEN_DIM=$8
ROT_DIM=$9
PRO_NUM=${10}
THRED=${11}
GAMMA=${12}
ALPHA=${13}
LEARNING_RATE=${14}
MAX_STEPS=${15}
WARM_STEPS=${16}
TEST_BATCH_SIZE=${17}
REG=${18}
ANCHOR_SIZE=${19}
DROP_SIZE=${20}
SAMPLE_ANCHORS=${21}
INIT_STD=${22}
LR_KGE=${23}
LOSS_TEMP=${24}

echo "Start Training......"

CUDA_VISIBLE_DEVICES=$GPU_DEVICE python -u $CODE_PATH/run_stage2.py --do_train \
    --cuda \
    --do_valid \
    --do_test \
    --data_name $FULL_DATA_PATH \
    --model $MODEL \
    -n $NEGATIVE_SAMPLE_SIZE -b $BATCH_SIZE -d $HIDDEN_DIM \
    -hd $ROT_DIM -dn $PRO_NUM -th $THRED \
    -g $GAMMA -a $ALPHA -adv \
    -lr $LEARNING_RATE --max_steps $MAX_STEPS --warm_up_steps $WARM_STEPS \
    -save $SAVE --test_batch_size $TEST_BATCH_SIZE \
    --drop_size $DROP_SIZE --anchor_size $ANCHOR_SIZE --sample_anchors $SAMPLE_ANCHORS --lr_kge $LR_KGE \
    -r $REG --init_std $INIT_STD \
    -t $LOSS_TEMP > ./$LOG.log 2>&1 &