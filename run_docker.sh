WORKSPACE=/media/data/yolov7
DATA=/media/data/datasets

docker run -it --rm \
	--gpus all \
    -w $WORKSPACE \
	-v $WORKSPACE:$WORKSPACE \
	-v $DATA:$DATA \
	--shm-size=64g \
	yolov7

	# -v $HOME/.Xauthority:/root/.Xauthority:rw \
	# -v /tmp/.X11-unix:/tmp/.X11-unix \
	# -e DISPLAY=unix$DISPLAY \
	# -e QT_X11_NO_MITSHM=1 \
	# --net host \
