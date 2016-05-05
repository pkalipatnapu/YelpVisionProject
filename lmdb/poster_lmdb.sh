DATA=/home/ubuntu/Yelp/data/
TOOLS=/home/ubuntu/caffe/build/tools

TEST_DATA_ROOT=/home/ubuntu/Yelp/data/poster_photos/

RESIZE=true
if $RESIZE; then
  RESIZE_HEIGHT=256
  RESIZE_WIDTH=256
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi

if [ ! -d $DATA/poster_lmdb ]; then
  GLOG_logtostderr=1 $TOOLS/convert_imageset \
      --resize_height=$RESIZE_HEIGHT \
      --resize_width=$RESIZE_WIDTH \
      $TEST_DATA_ROOT \
      $DATA/poster.txt \
      $DATA/poster_lmdb
  echo "Created poster lmdb."
fi

