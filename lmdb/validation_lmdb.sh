DATA=/home/ubuntu/Yelp/data/
TOOLS=/home/ubuntu/caffe/build/tools

VAL_DATA_ROOT=/home/ubuntu/Yelp/data/train_photos/

RESIZE=true
if $RESIZE; then
  RESIZE_HEIGHT=256
  RESIZE_WIDTH=256
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi

if [ ! -d $DATA/val_lmdb ]; then
  GLOG_logtostderr=1 $TOOLS/convert_imageset \
      --resize_height=$RESIZE_HEIGHT \
      --resize_width=$RESIZE_WIDTH \
      $VAL_DATA_ROOT \
      $DATA/val.txt \
      $DATA/val_lmdb
  echo "Created validation lmdb."
fi

