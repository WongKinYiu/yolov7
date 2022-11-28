#!/bin/bash
# Checks if gdown is installed
if ! type "gdown" > /dev/null; then
    read -p "Install gdown (google drive downloader) with pip3 (y/n)? " yn
    case $yn in
        [Yy]* ) pip3 install gdown;;
        * ) echo "Not installing gdown. Exiting.."; exit;;
    esac
fi

for arch in $@
do
    case $arch in
        yolov7-tiny)
            gdown -O yolov7-tiny_state.pt "https://drive.google.com/uc?id=1GYzWblAU8IAiydMOl1ND8ppgnf85knXS&confirm=t";;
        yolov7)
            gdown -O yolov7_state.pt "https://drive.google.com/uc?id=1er9yZV2Yeep0JjC7JhB86-7iC6nWUy1F&confirm=t";;
        yolov7x)
            gdown -O yolov7x_state.pt "https://drive.google.com/uc?id=1m_DGALC4XjpEW5bvt31wAFt33H-5jOto&confirm=t";;
        yolov7-w6)
            gdown -O yolov7-w6_state.pt "https://drive.google.com/uc?id=1RaRZ5uFJPnvDrVFxiOaYbpzGajej7i3Z&confirm=t";;
        yolov7-e6)
            gdown -O yolov7-e6_state.pt "https://drive.google.com/uc?id=1F_17FmnBwxm7ymGObdkD3JIZYv5fDiPU&confirm=t";;
        yolov7-d6)
            gdown -O yolov7-d6_state.pt "https://drive.google.com/uc?id=1Q2Uwr7S0bpKvnObbgpkAYvkNME_W3viW&confirm=t";;
        yolov7-e6e)
            gdown -O yolov7-e6e_state.pt "https://drive.google.com/uc?id=1Dibo8dxQuReyfN9h_3UUgI5Wd_rADOk5&confirm=t";;
        *)
            echo $arch weights are not available;;
    esac
done
