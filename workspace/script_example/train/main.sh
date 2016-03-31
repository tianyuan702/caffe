#!/usr/bin/env sh

TOOLS=build/tools
PROJ_DIR=workspace/train_bird

# caffe origin
#MODEL_PATH=/home/htc/workspace/cnn_work/model/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel
#SOLVER_PATH=${PROJ_DIR}/script/train/caffenet_origin/solver_origin.prototxt
#LOG_DIR=${PROJ_DIR}/log/caffenet
#SNAPSHOT_DIR=${PROJ_DIR}/snapshot/caffenet
#LOG_FILE=$LOG_DIR/log_fintune_origin.txt

# caffe bilinear 1
#MODEL_PATH=/home/htc/workspace/cnn_work/model/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel
#SOLVER_PATH=${PROJ_DIR}/script/train/caffenet_bilinear/solver_bilinear_1.prototxt
#LOG_DIR=${PROJ_DIR}/log/caffenet
#SNAPSHOT_DIR=${PROJ_DIR}/snapshot/caffenet
#LOG_FILE=$LOG_DIR/log_fintune_bilinear_1.txt

MODEL_PATH=model/bvlc_googlenet_bn/googlenet_bn_stepsize_6400_iter_1200000.caffemodel
SOLVER_PATH=${PROJ_DIR}/script/train/bvlc_googlenet_bn/solver.prototxt
LOG_DIR=${PROJ_DIR}/log/bvlc_googlenet_bn
SNAPSHOT_DIR=${PROJ_DIR}/snapshot/bvlc_googlenet_bn
LOG_FILE=$LOG_DIR/finetune.txt

#MODEL_PATH=/mnt_data/data/yuan_tian/model/vgg/VGG_ILSVRC_16_layers.caffemodel
#SOLVER_PATH=${PROJ_DIR}/script/train/solver_vgg.prototxt
#LOG_DIR=${PROJ_DIR}/log/vgg
#SNAPSHOT_DIR=${PROJ_DIR}/snapshot/vgg

if [ ! -d "$LOG_DIR" ]; then
  mkdir -p "$LOG_DIR"
fi
if [ ! -d "$SNAPSHOT_DIR" ]; then
  mkdir -p "$SNAPSHOT_DIR"
fi

GLOG_logtostderr=1 $TOOLS/caffe train --solver=$SOLVER_PATH --weights=$MODEL_PATH 2>&1 | tee $LOG_FILE
