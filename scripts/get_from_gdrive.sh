#!/bin/bash
# Download command: bash scripts/get_from_gdrive.sh

# Settings (there can be no space after the '=')
url=https://drive.google.com/file/d/1BAVSo4oTvzmiBvPrUsCVfhdS4JSqWLeO/view?usp=share_link
FILENAME=small_set.zip

# Extract ID
FILEID=${url%/*}  # retain the part before the last slash
FILEID=${FILEID##*/}  # retain the part after the second to last slash

# # Download / Unzip / Delete zip
echo 'Downloading' $FILENAME ' ...'
mkdir practice_data/${FILENAME%.*}
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=$FILEID' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$FILEID" -O $FILENAME && rm -rf /tmp/cookies.txt  
unzip -q $FILENAME -d practice_data/${FILENAME%.*}/
rm -f $FILENAME

# Make YOLO folder and convert to format
echo 'Converting to YOLO format'
mkdir practice_data/${FILENAME%.*}/yolo
python3 convert_tgsse_to_yolo_format.py --data_path practice_data/${FILENAME%.*} --save_path practice_data/${FILENAME%.*}/yolo

# Get weights
echo 'Downloading weights'
bash scripts/get_weights.sh

echo 'Done (if nothing shows in VS code, please refresh file explorer)'


