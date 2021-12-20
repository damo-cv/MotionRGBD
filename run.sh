#!/bin/sh
TRAIN=$1
CONFIG=$2
GPUID=$3
GPUNUM=$4
PORT=${PORT:-29509}
CUDA_VISIBLE_DEVICES=$GPUID python -m torch.distributed.launch --nproc_per_node=$GPUNUM --master_port=$PORT $TRAIN --config $CONFIG --nprocs $GPUNUM --save_output
