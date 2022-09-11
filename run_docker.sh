WORKSPACE=/media/data/yolov7
DATA=/media/data/datasets

docker run -it --rm \
	--gpus all \
    -w $WORKSPACE \
	-v $WORKSPACE:$WORKSPACE \
	-v $DATA:$DATA \
	yolov7
