import numpy as np
import os
import pandas as pd
from pathlib import Path

from config import get_parser
from FCG import FCG, generate_df_tracking, generate_df_tracking_with_feats


def save_results(df, f_name):
    # Save results in a format that can be ingested by evaluation script
    # frame, ID, x_left, y_top, w, h, -1, -1, -1, -1
    # 583,71,543.09,460.61,22.38,68.97,-1,-1,-1,-1
    os.makedirs(f_name.parent, exist_ok=True)
    df_sorted = df.sort_values(by=['ID', 'frame'])
    frame_list = df_sorted['frame'].to_list()
    id_list = df_sorted['ID'].to_list()
    x_left = [bbox[0] for bbox in df_sorted['bbox'].to_list()]
    y_top = [bbox[1] for bbox in df_sorted['bbox'].to_list()]
    w = [bbox[2] for bbox in df_sorted['bbox'].to_list()]
    h = [bbox[3] for bbox in df_sorted['bbox'].to_list()]
    res_dict = {'frame': frame_list, 'ID': id_list, 'x_left': x_left, 'y_top': y_top, 'w': w, 'h': h}
    df_save = pd.DataFrame(data=res_dict)
    df_save['extra1'] = -1
    df_save['extra2'] = -1
    df_save['extra3'] = -1
    df_save['extra4'] = -1
    df_save.to_csv(str(f_name), header=False, index=False)  # As an example


def some_tracking(df, frames_window=6, seq='debug'):
    # Initialize FCG with parameters
    fcg = FCG(df, frames_window=frames_window, iou_off=MAX_IOU_TH,
              opt_iou=FLAG_IOU, opt_motion=FLAG_MOTION, opt_temp=FLAG_TEMP, opt_spatial=FLAG_SPATIAL,
              max_prop=MAX_PROP, opt_randorder=FLAG_RANDORDER, opt_noncont=FLAG_NONCONT)

    # Build tracklets
    dist_thresh = 0.055
    df_result, linkage_matrices = fcg.generate_tracklets(df, thresh=dist_thresh)

    # Stack the results based on the lifted frames and the cluster IDs
    df_cluster = df_result.groupby(['frame_cluster', 'cluster_ID'], as_index=False).agg({
        'frame': lambda x: list(x), 'bbox': [np.stack], 'feats': [lambda x: np.median(np.stack(x), axis=0), np.stack], 'prob': lambda x: list(x)})

    # Change column names
    df_cluster.columns = ['frame', 'cluster_ID', 'frame_list', 'bbox', 'feats', 'feats_stack', 'prob']

    # Combine tracklets within the first level lifted frames
    df_result, linkage_matrices = fcg.fuse_lifted_frames(df_cluster, window=W_FUSE, thresh=dist_thresh)

    # Loop until last level of the hierarchy
    while True:
        # Stack results
        df_cluster = df_result.groupby(['frame_cluster', 'cluster_ID'], as_index=False).agg({
            'frame_list': [np.hstack], 'bbox': [np.vstack], 'feats': [lambda x: np.median(np.stack(x), axis=0)], 'feats_stack': [lambda x: np.vstack(x)], 'prob': np.hstack})

        # Rename columns
        df_cluster.columns = ['frame', 'cluster_ID', 'frame_list', 'bbox', 'feats', 'feats_stack', 'prob']

        # Break if all tracklets are within the same lifted frame
        if all(df_cluster['frame'].unique() == 1):
            break

        # In case there are still multiple lifted frames continue fusing tracklets
        df_result, linkage_matrices = fcg.fuse_lifted_frames(df_cluster, window=W_FUSE, thresh=dist_thresh)

    # For analysing the features we can save them into a pickle file
    if FLAG_SAVE_FEATS:
        df_tracking_feats = generate_df_tracking_with_feats(df_cluster)
        res_path = Path(f'results/{DATASET}/{DATASET}-{MODE}/{TRACKER_ID}/feats')
        os.makedirs(str(res_path), exist_ok=True)
        fname = res_path / f'{seq}.pkl'
        pd.to_pickle(df_tracking_feats, fname)

    # Generate clean dataframe containing all the tracking information
    df_tracking = generate_df_tracking(df_cluster)

    return df_tracking


def main():
    # Read sequences from path
    seqs = sorted([seq.name for seq in GT_PATH.iterdir() if seq.is_dir()])  # seqs = ['MOT17-12']

    if DATASET == 'mot17':
        # MOT17-train for TrackEval to recognize the folder
        if FPS_RATIO == 1:
            res_path = Path(f'results/mot17/MOT17-train/{TRACKER_ID}/data')
        else:
            res_path = Path(f'results/mot17/MOT17_{FPS_RATIO}-train/{TRACKER_ID}/data')
    elif DATASET == 'mot20':
        if FPS_RATIO == 1:
            res_path = Path(f'results/mot20/MOT20-train/{TRACKER_ID}/data')
        else:
            res_path = Path(f'results/mot20/MOT20_{FPS_RATIO}-train/{TRACKER_ID}/data')
    elif DATASET == 'dancetrack':
        if FPS_RATIO == 1:
            res_path = Path(f'results/dancetrack/dancetrack-{MODE}/{TRACKER_ID}/data')
        else:
            res_path = Path(f'results/dancetrack/dancetrack_{FPS_RATIO}-{MODE}/{TRACKER_ID}/data')
    else:
        print('Incorrect dataset, exiting')
        exit()

    print(f'Saving to {str(res_path)}')

    # Make paths in case they don't exist
    os.makedirs(str(res_path), exist_ok=True)

    # Tracking for each sequence
    for seq in seqs:
        # Read pickle file containing detections and features
        print(f'{seq}')
        if FPS_RATIO == 1:
            feat_file = f'{FEAT_PATH}/{seq}-feats_{DETECTOR}_{REID}.pkl'
        else:
            seq_elems = seq.split('-')
            seq_mod = f'{seq_elems[0]}_{FPS_RATIO}-{seq_elems[1]}'
            feat_file = f'{FEAT_PATH}/{seq_mod}-feats_{DETECTOR}_{REID}.pkl'

        if not Path(feat_file).is_file():
            continue

        # Read pickle file with all the info (detections + feats)
        df = pd.read_pickle(feat_file)

        # Threshold detections based on probability
        if 'prob' in df.columns:
            thresh = 0.7
            mask_thresh = df['prob'] > thresh
            df = df[mask_thresh]
        else:
            # To make it consistent for saving the tracking with features for analysis
            df['prob'] = 1

        # Do the tracking
        df_tracking = some_tracking(df, frames_window=W_TRACKLET, seq=seq)

        print(f"Number of estimated tracks: {len(df_tracking['ID'].unique())}")

        # Replicate results needed for private detections
        if DATASET == 'mot17':
            fname = res_path / f'{seq}-DPM.txt'
            save_results(df_tracking, fname)
            fname = res_path / f'{seq}-FRCNN.txt'
            save_results(df_tracking, fname)
            fname = res_path / f'{seq}-SDP.txt'
            save_results(df_tracking, fname)
        elif DATASET == 'mot20':
            fname = res_path / f'{seq}.txt'
            save_results(df_tracking, fname)
        else:
            fname = res_path / f'{seq}.txt'
            save_results(df_tracking, fname)


if __name__ == '__main__':
    # Check for objects leaving the scene in IoU calculation
    parser = get_parser()
    args = parser.parse_args()
    # PATHS
    MODE = args.mode
    GT_PATH = Path(args.datapath) / MODE  # Path-with-frames/train
    FEAT_PATH = Path(args.feat)
    # Variables from argparse
    DATASET = args.dataset
    DETECTOR = args.detector
    REID = args.reid
    MAX_IOU_TH = args.max_iou_th
    W_TRACKLET = args.w_tracklet
    W_FUSE = args.w_fuse
    MAX_PROP = args.max_prop
    FPS_RATIO = args.fps_ratio  # For tracking experiments with lower fps dataset
    FLAG_SAVE_FEATS = args.save_feats
    FLAG_IOU = args.iou
    FLAG_TEMP = args.temp
    FLAG_SPATIAL = args.spatial
    FLAG_MOTION = args.motion
    FLAG_RANDORDER = args.randorder
    FLAG_NONCONT = args.noncont

    if args.mod is None:
        TRACKER_ID = f'fcg-{DETECTOR}-{REID}'
    else:
        TRACKER_ID = f'fcg-{DETECTOR}-{REID}-{args.mod}'

    main()
