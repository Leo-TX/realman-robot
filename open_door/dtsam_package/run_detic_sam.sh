IMAGE_PATH="/media/datadisk10tb/leo/projects/realman-robot/open_door/images/image1/rgb.png"
CLASSES="switch"
DEVICE="cuda:1"
THRESHOLD=0.3
python detic_sam.py -i "$IMAGE_PATH" -c "$CLASSES" -d "$DEVICE" -t "$THRESHOLD"