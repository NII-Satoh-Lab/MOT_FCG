import argparse


def get_parser():
    parser = argparse.ArgumentParser(description='Face detection and classification for politicians in Japanese TV.')

    # Important configuration variables
    parser.add_argument('--dataset', type=str, default='mot17', help='Mode name for saving files.')
    parser.add_argument('--mode', default='train', type=str, help='train or test.')
    parser.add_argument('--detector', type=str, default='YOLOX', help='Detector to be used. FRCNN, SDP, Bresee, SGT, YOLOX, GT.')
    parser.add_argument('--reid', type=str, default=None, help='Reidentification model to be used. SBS, MGN.')
    parser.add_argument('--mod', type=str, default=None, help='Tracker name modifier to do testing of features.')

    # Paths
    parser.add_argument('--datapath', type=str, default='datasets/MOT17Det', help='Dataset path with frames inside.')
    parser.add_argument('--feat', type=str, default='feats', help='Features files path.')

    # Tracking-specific configuration variables
    parser.add_argument('--max_iou_th', type=float, default=0.15, help='Max value to multiply the distance of two close objects.')
    parser.add_argument('--w_tracklet', type=int, default=10, help='Window size per tracklet')
    parser.add_argument('--w_fuse', type=int, default=3, help='Window size per fusion in hierarchy')
    parser.add_argument('--max_prop', type=int, default=10000, help='Difficult the fusion when the frame difference is larger than this value.')
    parser.add_argument('--fps_ratio', type=int, default=1, help='Use lower fps dataset if lower than 1.')

    # Flags
    parser.add_argument('--save_feats', action='store_true', help='Save tracking + feature vectors as pkl file for analysis.')
    parser.add_argument('--iou', action='store_true', help='Add IoU distance to further improve the tracker.')
    parser.add_argument('--temp', action='store_true', help='Use temporal distance to further improve the tracker.')
    parser.add_argument('--spatial', action='store_true', help='Use spatial distance to further improve the tracker.')
    parser.add_argument('--motion', action='store_true', help='Add motion estimation to further improve the tracker.')
    parser.add_argument('--randorder', action='store_true', help='Random order of lifted frames for testing.')
    parser.add_argument('--noncont', action='store_true', help='Do not enforce continuous clustering. Allow all tracklets to cluster with whoever they want.')

    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
