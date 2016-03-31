#!/usr/bin/env sh
# Create the imagenet lmdb inputs
# N.B. set the path to the imagenet train + val data dirs

PROJ_DIR=workspace/train_bird
TOOLS=build/tools
DATA_ROOT=/media/htc/work/dataset/CUB_200_2011/CUB_200_2011/images
#DATA_ROOT=$PROJ_DIR/crop_img
LOG_DIR=${PROJ_DIR}/log

# Set RESIZE=true to resize the images to 256x256. Leave as false if images have
# already been resized using another tool.
RESIZE=true
if $RESIZE; then
  RESIZE_HEIGHT=256
  RESIZE_WIDTH=256
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi
mkdir -p $LOG_DIR
echo "Creating train lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $DATA_ROOT \
    $PROJ_DIR/train_list.txt \
    $PROJ_DIR/bird_train_lmdb 2>&1 | tee $LOG_DIR/log_create_train.txt

echo "Creating val lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $DATA_ROOT \
    $PROJ_DIR/test_list.txt \
    $PROJ_DIR/bird_val_lmdb 2>&1 | tee $LOG_DIR/log_create_test.txt

echo "Done."
