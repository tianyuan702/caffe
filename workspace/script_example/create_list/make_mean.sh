#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

PROJ_DIR=/mnt_data/data/yuan_tian/train_bird
TOOLS=/mnt_data/data/yuan_tian/caffe_root/caffe/build/tools

$TOOLS/compute_image_mean $PROJ_DIR/bird_train_lmdb \
  $PROJ_DIR/bird_mean.binaryproto

echo "Done."
