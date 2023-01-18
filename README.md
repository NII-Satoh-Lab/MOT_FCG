# Multiple Object Tracking from appearance by hierarchically clustering tracklets

Presented as a spotlight in BMVC22

[**Multiple Object Tracking from appearance by hierarchically clustering tracklets**](https://arxiv.org/abs/2210.03355)

## Tracking performance
### Results on test set of different datasets
| Dataset    | HOTA | DetA | AssA | MOTA | IDF1 |
|------------|------|------|------|------|------|
| [MOT17](https://motchallenge.net/) | 62.6 | 62.2 | 63.4 | 76.7 | 77.7 |
| [MOT20](https://motchallenge.net/) | 57.3 | 56.7 | 58.1 | 68.0 | 69.7 |
| [DanceTrack](https://github.com/DanceTrack/DanceTrack) | 48.7 | 79.8 | 29.9 | 89.9 | 46.5 |

## Installation
*Code tested in python3.8*
### Download github repository and position in the ROOT folder
```
git clone https://github.com/NII-Satoh-Lab/MOT_FCG.git
```

```
cd $YOUR_PATH/MOT_FCG
```

### Install dependencies
```
pip install -r requirements.txt
```

## Data preparation

1. Download [MOT17](https://motchallenge.net/), [MOT20](https://motchallenge.net/), and [DanceTrack](https://github.com/DanceTrack/DanceTrack).

2. Download detections and feature data (around 20GB)

*Note: This scripts downloads and prepares detections and features for MOT17, MOT20, and dancetrack. If you want to download only a specific dataset modify the below script.*

```
bash scripts/prepare_data.sh
```

## Run example
Run the code example for MOT17 using YOLOX and SBS features

```
python src/fcg_tracking.py --gt $YOUR_PATH_TO_DATASETS/MOT17Det --dataset mot17 --mode train --detector YOLOX --reid SBS --temp --iou --spatial --motion --w_tracklet 6 --w_fuse 3 --max_prop 40
```

## Evaluation

### Download evaluation repository from [TrackEval](https://github.com/JonathonLuiten/TrackEval)

Download the git repository and follow the installation instructions and the ground truth data positioning.

```
git clone https://github.com/JonathonLuiten/TrackEval
```

```
python $TRACKEVAL_PATH/scripts/run_mot_challenge.py --TRACKERS_FOLDER $YOUR_PATH/MOT_FCG/results/mot17 --BENCHMARK MOT17 --TRACKERS_TO_EVAL fcg-YOLOX-SBS --GT_FOLDER $TRACKEVAL_PATH/data/MOT17 --SPLIT_TO_EVAL train --METRICS HOTA CLEAR Identity --PLOT_CURVES False --PRINT_CONFIG False --TIME_PROGRESS False
```

## Citation

```
@inproceedings{Girbau_2022_BMVC,
author    = {Andreu Girbau and Ferran Marques and Shin'ichi Satoh},
title     = {Multiple Object Tracking from appearance by hierarchically clustering tracklets},
booktitle = {33rd British Machine Vision Conference 2022, {BMVC} 2022, London, UK, November 21-24, 2022},
publisher = {{BMVA} Press},
year      = {2022},
url       = {https://bmvc2022.mpi-inf.mpg.de/0362.pdf}
}
```


## TODO
- [x] Code with our extracted detections + features
- [x] Bibcitation
- [ ] Script to generate visualizations (e.g. videos)
- [ ] Scripts to generate low fps MOT17 videos
