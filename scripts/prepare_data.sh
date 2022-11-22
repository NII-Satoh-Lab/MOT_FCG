#!/usr/bin/env bash

# Make directory (run it from MOT_FCG)
mkdir "feats"

# Download the detections and features from the webpage
wget --no-check-certificate www.satoh-lab.nii.ac.jp/member/agirbau/tracking/files/MOT17.zip -P feats
wget --no-check-certificate www.satoh-lab.nii.ac.jp/member/agirbau/tracking/files/MOT20.zip -P feats
wget --no-check-certificate www.satoh-lab.nii.ac.jp/member/agirbau/tracking/files/dancetrack.zip -P feats

# Unzip files
unzip feats/MOT17.zip -d feats
unzip feats/MOT20.zip -d feats
unzip feats/dancetrack.zip -d feats

# Move files to "feats folder"
mv feats/MOT17/* feats/.
mv feats/MOT20/* feats/.
mv feats/dancetrack/* feats/.

# Remove dirs
rmdir feats/MOT17
rmdir feats/MOT20
rmdir feats/dancetrack

# Remove zip files
rm feats/MOT17.zip
rm feats/MOT20.zip
rm feats/dancetrack.zip