# Create an lmdb from training and validation images.
# This will fail if the lmdb already exists.
DATA=/home/ubuntu/Yelp/data/
TOOLS=/home/ubuntu/caffe/build/tools

TRAIN_DATA_ROOT=/home/ubuntu/Yelp/data/train_photos/

RESIZE=true
if $RESIZE; then
  RESIZE_HEIGHT=256
  RESIZE_WIDTH=256
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi

if [ ! -d $DATA/train_lmdb ]; then
    # Do no shuffle, since we will extract labels seperately.
  GLOG_logtostderr=1 $TOOLS/convert_imageset \
      --resize_height=$RESIZE_HEIGHT \
      --resize_width=$RESIZE_WIDTH \
      $TRAIN_DATA_ROOT \
      $DATA/train.txt \
      $DATA/train_lmdb
  echo "Created train lmdb."
fi

