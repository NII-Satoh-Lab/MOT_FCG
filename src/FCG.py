# Class for the FCG (Feature Combinational Grouping)
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import scipy.spatial.distance as distance

from config import get_parser
from mot_utils import iou_dist, iou_dist_clust, get_same_frames_idx, compute_frames_distance, bbox_dist, bbox_dist_clust


class FCG:
    def __init__(self, df, frames_window, **kwargs):
        # Data to process
        self.df_original = df
        self.frames_window = frames_window if frames_window > 1 else max(df['frame'])
        # Variables
        self.linkage_matrix = None
        self.cluster_window = None
        self.tree_level = 1
        self.iou_off = kwargs.get('iou_off', 0.15)
        self.opt_iou = kwargs.get('opt_iou', False)
        self.opt_temp = kwargs.get('opt_temp', False)
        self.opt_spatial = kwargs.get('opt_spatial', False)
        self.opt_motion = kwargs.get('opt_motion', False)
        self.opt_randorder = kwargs.get('opt_randorder', False)
        self.opt_noncont = kwargs.get('opt_noncont', False)
        self.max_prop = kwargs.get('max_prop', 10000)

    def generate_tracklets(self, df, window=None, thresh=0.055, method='average'):
        # In this function we generate tracklets from video frames
        df_window_list = []
        cluster_id_list = []
        new_frame_cluster_list = []
        linkage_matrix_list = []
        max_cluster_id = 0
        frame_cluster = 1
        max_frame = max(df['frame'])

        # Gather the frames indexes in temporal windows for the whole video
        frames_window = self.frames_window if window is None else window
        frames_to_visit = np.arange(frames_window, max_frame, frames_window)
        frames_to_visit = np.append(frames_to_visit, max_frame)

        # Test (shuffle in order to check that order matters)
        if self.opt_randorder:
            np.random.shuffle(frames_to_visit)

        # Group all frames in temporal windows and generate tracklets
        visited_frames = []
        for frame in frames_to_visit:
            # Filter df by frame window
            mask_window = (df['frame'] > frame-frames_window) & (df['frame'] <= frame) & (~df['frame'].isin(visited_frames))
            df_window = df[mask_window]
            if df_window.empty:
                continue

            # Get features for the temporal window
            X_feats = np.asarray(df_window['feats'].to_list()).squeeze(1)
            X_frames = np.expand_dims(np.asarray(df_window['frame'].to_list()), axis=1)
            X_bboxes = df_window['bbox'].to_list()

            # Distance between object features
            dist_feats = distance.pdist(X_feats, 'cosine')

            # Compute distance between frames to have the index where the same frames appear to not to match elements from the same frame.
            same_frames_idx = distance.pdist(X_frames, 'euclidean') == 0
            dist_iou_square = iou_dist(X_bboxes, X_frames)
            dist_bbox_square = bbox_dist(X_bboxes, X_frames)

            # Compute close (iou) and far distances (dist bbox)
            dist_iou = distance.squareform(dist_iou_square)
            dist_bbox = distance.squareform(dist_bbox_square)

            # Compute the overall distance between objects.
            dist_matrix = dist_feats.copy()

            # Options for ablation
            # If an object is really close to another one make it easier to fuse (but not 0)
            if self.opt_iou:
                dist_iou += self.iou_off
                mask_iou = dist_iou > 1
                dist_iou[mask_iou] = 1
                dist_matrix = dist_matrix * dist_iou

            # Consider far distances
            if self.opt_spatial:
                dist_spat_norm_far = distance.squareform(self.normalize_spatial_dist_far(distance.squareform(dist_bbox)))
                dist_matrix = dist_matrix * dist_spat_norm_far

            # Don't allow elements from the same frame to fuse
            dist_matrix[same_frames_idx] += 1000
            # v_debug = distance.squareform(dist_matrix)

            # Build hierarchical tree for the specified window and cluster based on the hierarchical tree
            linkage_matrix = linkage(dist_matrix, metric='cosine', method=method)
            cluster_window = fcluster(linkage_matrix, t=thresh, criterion='distance', depth=self.frames_window) + max_cluster_id

            # Assign IDs to the clusters and frames within the same window
            max_cluster_id = max(cluster_window)
            df_window_list.append(df_window)
            cluster_id_list += list(cluster_window)
            [new_frame_cluster_list.append(frame_cluster) for _ in cluster_window]
            linkage_matrix_list.append(linkage_matrix)
            visited_frames += list(df_window['frame'].unique())

            # Assign new frame IDs to clusters
            frame_cluster += 1

        # Put everything in a dataframe to be fused in the lifted frames
        df_cluster = pd.concat(df_window_list)
        df_cluster['cluster_ID'] = cluster_id_list
        df_cluster['frame_cluster'] = new_frame_cluster_list

        return df_cluster, linkage_matrix_list

    def fuse_lifted_frames(self, df, window=None, thresh=0.055, method='average'):
        # Here we are fusing tracklets from lifted frames
        df_window_list = []
        cluster_id_list = []
        new_frame_cluster_list = []
        linkage_matrix_list = []
        frame_cluster = 1

        # Lifted frames index from previous lifted frame ID ('frame' parameter reused)
        lifted_frames = df['frame'].unique()

        # To test random order (in contrast with sequential order)
        if self.opt_randorder:
            np.random.shuffle(lifted_frames)

        # Group continuous lifted frames to be fused
        # e.g. [[1, 2, 3], [4, 5, 6], ... [N-2, N-1, N]]
        n = 0
        lifted_frames_groups = []
        l_frame_group = []
        for l_frame in lifted_frames:
            if n == window:
                lifted_frames_groups.append(l_frame_group)
                l_frame_group = []
                n = 0
            l_frame_group.append(l_frame)
            n += 1

        # Add last group
        if len(l_frame_group) > 0:
            lifted_frames_groups.append(l_frame_group)

        # By default, fuse only the first group of frames (to enforce sequentiality in fusion)
        if not self.opt_noncont:
            lifted_frames_groups = [lifted_frames_groups[0]]

        # Fuse tracklets from grouped lifted frames
        max_cluster_id = max(df['cluster_ID']) + 1  # Cluster IDs for the lifted frames
        visited_frames = []

        for lifted_frames in lifted_frames_groups:
            # Filter df to contain the lifted frames within a temporal window
            mask_window = df['frame'].isin(lifted_frames)
            df_window = df[mask_window]

            # Get features for the temporal window
            X_feats = np.asarray(df_window['feats'].to_list()).squeeze(1)
            X_frames = np.expand_dims(np.asarray(df_window['frame'].to_list()), axis=1)
            X_bboxes = df_window['bbox'].to_list()

            # Distance between tracklets (median appearance feature)
            dist_feats = distance.pdist(X_feats, 'cosine')

            # Be sure to be able to fuse tracks that correspond to the same lifted frame, but not share any raw frame
            X_frames_list = df_window['frame_list'].to_list()
            same_frames_idx = get_same_frames_idx(X_frames, X_frames_list)
            same_frames_idx = distance.squareform(same_frames_idx)
            dist_iou_square = iou_dist_clust(X_bboxes, X_frames, opt_motion=self.opt_motion)
            dist_frames_square = compute_frames_distance(X_frames, X_frames_list)
            dist_bbox_square = bbox_dist_clust(X_bboxes, X_frames, opt_motion=self.opt_motion)

            # Compute close-distance using IoU and bbox distances for far ones
            dist_iou = distance.squareform(dist_iou_square)  # distance.squareform returns a condensed array if a squareform is passed and vice-versa
            dist_bbox = distance.squareform(dist_bbox_square)

            # Compute the overall distance between objects
            dist_matrix = dist_feats.copy()

            # Options for ablation
            # If an object is really close to another one make them easier to fuse (but not 0)
            # Also, take into account how far temporally they are (problems with objects leaving the scene)
            if self.opt_iou:
                dist_iou += self.iou_off
                mask_iou = dist_iou > 1
                dist_iou[mask_iou] = 1
                dist_matrix = dist_matrix * dist_iou

            # Final tracklet distance from spatial and temporal distances
            # Consider far distances
            if self.opt_spatial:
                dist_spat_norm_far = distance.squareform(self.normalize_spatial_dist_far(distance.squareform(dist_bbox)))
                dist_matrix = dist_matrix * dist_spat_norm_far

            # Consider temporal distance between video frames
            if self.opt_temp:
                dist_temp_norm = self.normalize_temporal_dist(distance.squareform(dist_frames_square))
                dist_matrix = dist_matrix * dist_temp_norm

            # Use 0.1 for visualization purposes.
            dist_matrix[same_frames_idx] += 1000
            # v_debug = distance.squareform(dist_matrix)

            # Build hierarchical tree for the specified window and cluster based on the hierarchical tree
            linkage_matrix = linkage(dist_matrix, metric='cosine', method=method)
            cluster_window = fcluster(linkage_matrix, t=thresh, criterion='distance', depth=self.frames_window) + max_cluster_id

            # Assign IDs to the clusters and lifted frames within the same window
            max_cluster_id = max(cluster_window)
            df_window_list.append(df_window)
            cluster_id_list += list(cluster_window)
            [new_frame_cluster_list.append(frame_cluster) for _ in cluster_window]

            linkage_matrix_list.append(linkage_matrix)
            visited_frames += list(df_window['frame'].unique())

            # Assign new frame IDs to tracklet clusters from lifted frames
            frame_cluster += 1

        # For ablation (ignore sequentiality)
        if not self.opt_noncont:
            # Rest of the non visited frames
            mask = ~df['frame'].isin(visited_frames)
            df2 = df[mask]
            df_window_list.append(df2)
            cluster_id_list += list(df2['cluster_ID'].unique())
            non_visited_frames = [f[1] for f in df2['frame'].items()]
            new_frame_cluster_list += non_visited_frames

        # Put everything in a dataframe to be fused in the posterior lifted frames and climb one level of the hierarchy
        df_cluster = pd.concat(df_window_list)
        df_cluster['cluster_ID'] = cluster_id_list
        df_cluster['frame_cluster'] = new_frame_cluster_list
        self.tree_level += 1

        return df_cluster, linkage_matrix_list

    def normalize_temporal_dist(self, dist_frames_square):
        # Normalize to a constant that will penalize farther frames
        off_frames_square = np.ones_like(dist_frames_square)
        mask_start = dist_frames_square > self.max_prop

        # From 3 to 5
        start_score = 5
        off_frames_square[mask_start] = start_score

        mask_diag = dist_frames_square == 0  # Diagonal must be zero
        off_frames_square[mask_diag] = 0

        return off_frames_square

    @staticmethod
    def normalize_spatial_dist_far(dist_bbox_square):
        # Convert to a multiplier that will penalize farther bboxes
        off_spatial_square = np.ones_like(dist_bbox_square)

        # IoU: works with (2, 2)
        mask_far = dist_bbox_square >= 2
        off_spatial_square[mask_far] = 2  # Make 2 times more difficult to fuse

        mask_diag = dist_bbox_square == 0  # Diagonal must be zero
        off_spatial_square[mask_diag] = 0

        return off_spatial_square


def generate_df_tracking_with_feats(df_result):
    # Generate resulting df for tracking evaluation
    # Output:
    # frame, ID, bbox, prob, visibility, feats
    res_data = []
    # id_df_list = df_result['cluster_ID'].to_list()
    frames_df_list = df_result['frame_list'].to_list()
    bboxes_df_list = df_result['bbox'].to_list()
    feats_df_list = df_result['feats_stack'].to_list()
    prob_df_list = df_result['prob'].to_list()

    # The clustering doesn't provide an ordered set of tracks (track 1 in frame 450). Order them for the final results.
    first_frame = [f[0] if not isinstance(f, np.int64) else f for f in frames_df_list]
    first_frame_order = np.argsort(first_frame)
    # Assign ID 1 to some elements of the 1st frame instead of directly the clustering IDs
    # The elements that appear before in the sequence will have lower IDs than the latter.
    ordered_elements = [(new_id, frames_df_list[i], bboxes_df_list[i], feats_df_list[i], prob_df_list[i]) for new_id, i in enumerate(first_frame_order, start=1)]

    # for id, frames, bboxes in zip(id_df_list, frames_df_list, bboxes_df_list):
    for track_id, frames, bboxes, feats, probs in ordered_elements:
        # In case there's only 1 detection or several
        if isinstance(frames, np.int64):
            # frame, ID, bbox, prob, visibility, feats
            res_data.append([frames, track_id, bboxes[0], probs, -1, feats[0]])
        else:
            for frame, bbox, feat, prob in zip(frames, bboxes, feats, probs):
                # frame, ID, bbox, prob, visibility, feats
                res_data.append([frame, track_id, bbox, prob, -1, feat])

    df_tracking_feats = pd.DataFrame(data=res_data, columns=['frame', 'ID', 'bbox', 'prob', 'visibility', 'feats'])
    df_tracking_feats = df_tracking_feats.sort_values(by=['ID', 'frame'])
    return df_tracking_feats


def generate_df_tracking(df_result):
    # Generate resulting df for tracking evaluation
    # Output:
    # ID, frame, bbox
    res_data = []
    # id_df_list = df_result['cluster_ID'].to_list()
    frames_df_list = df_result['frame_list'].to_list()
    bboxes_df_list = df_result['bbox'].to_list()

    # The clustering doesn't provide an ordered set of tracks (track 1 in frame 450). Order them for the final results.
    first_frame = [f[0] if not isinstance(f, np.int64) else f for f in frames_df_list]
    first_frame_order = np.argsort(first_frame)
    # Assign ID 1 to some elements of the 1st frame instead of directly the clustering IDs
    # The elements that appear before in the sequence will have lower IDs than the latter.
    ordered_elements = [(new_id, frames_df_list[i], bboxes_df_list[i]) for new_id, i in enumerate(first_frame_order, start=1)]

    # for id, frames, bboxes in zip(id_df_list, frames_df_list, bboxes_df_list):
    for id, frames, bboxes in ordered_elements:
        # In case there's only 1 detection or several
        if isinstance(frames, np.int64):
            res_data.append([id, frames, bboxes[0]])
        else:
            for frame, bbox in zip(frames, bboxes):
                res_data.append([id, frame, bbox])

    df_tracking = pd.DataFrame(data=res_data, columns=['ID', 'frame', 'bbox'])
    return df_tracking


def plot_dendrogram(model, **kwargs):
    fig, ax1 = plt.subplots()
    dendrogram(model, **kwargs)
    plt.setp(ax1.get_xticklabels(), ha="right", rotation=60, size=6)
    plt.ylim(0, 0.5)
    plt.tight_layout()
    plt.show()


def main_toy_distances(id_test=2):
    # seqs = [seq.name for seq in GT_PATH.iterdir() if seq.is_dir()]
    seqs = ['MOT17-02']
    for seq in seqs:
        df = pd.read_pickle(f'{FEAT_PATH}/{seq}-feats_bytetrack_x_mot17.pkl')
        mask_id = df['ID'] == id_test
        df_id = df[mask_id]
        X = np.asarray(df_id['feats'].to_list()).squeeze(1)
        X_bboxes = df_id['bbox'].to_list()
        # To be able to increase the distance of elements inside the same frame compute it here
        dist_matrix = distance.pdist(X, 'cosine')
        dist_matrix_square = distance.squareform(dist_matrix)
        # Check modes of the samples
        v_mean = np.mean(X, axis=0)
        v_std = np.std(X, axis=0)


def main_toy():
    X1 = np.asarray([[0, 0], [0, 1], [1, 1]])
    X2 = np.asarray([[1, 0], [2, 0], [2, 2]])
    c1 = np.expand_dims(X1.sum(axis=0) / 3, axis=0)
    c2 = np.expand_dims(X2.sum(axis=0) / 3, axis=0)
    X1_c2 = np.append(X1, c2, axis=0)
    X2_c1 = np.append(X2, c1, axis=0)
    l1 = linkage(X1_c2, metric='euclidean', method='complete')
    l2 = linkage(X2_c1, metric='euclidean', method='complete')
    plot_dendrogram(l1)
    plot_dendrogram(l2)


def main():
    # seqs = [seq.name for seq in GT_PATH.iterdir() if seq.is_dir()]
    seqs = ['MOT17-02']
    for seq in seqs:
        df = pd.read_pickle(f'{FEAT_PATH}/{seq}-feats_bytetrack_x_mot17.pkl')
        fcg = FCG(df, frames_window=3, force_fuse='ID')
        df_result, linkage_matrices = fcg.generate_tracklets(df)
        plot_dendrogram(linkage_matrices[0], color_threshold=0.1)  # Example dendogram


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    GT_PATH = Path(args.datapath) / args.mode
    FEAT_PATH = Path(args.feat)
    main()
    # main_toy()
    # main_toy_distances(id_test=2)
