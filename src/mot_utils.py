import numpy as np
np.random.seed(123)


class ColorGenerator:
    def __init__(self):
        self.colors = None
        self.generate_colors()

    def __getitem__(self, index):
        return (int(self.colors[index][0]), int(self.colors[index][1]), int(self.colors[index][2]))

    def generate_colors(self):
        # Casting dtype=int doesn't change from int64
        c_r = np.linspace(1, 255, num=100, endpoint=True)
        c_g = np.linspace(1, 255, num=100, endpoint=True)
        c_b = np.linspace(1, 255, num=100, endpoint=True)

        colors = np.asarray(np.meshgrid(c_r, c_g, c_b), dtype=int).reshape(3, -1).T
        self.colors = np.random.permutation(colors)


def motion_constant_velocity(bboxes):
    # Get a set of bboxes as input and estimate the motion (velocity) in a "constant velocity assumption"
    if len(bboxes) > 1:
        v_bbox = bboxes[-1] - bboxes[-2]
        v = [v_bbox[0], v_bbox[1], 0, 0]
    else:
        v = 0

    return v


def compute_frames_distance(frames, frames_list):
    # Compute the distance between frames within lifted frames to avoid late-fusion of nodes that have nothing in common
    # Use of left-right (previous/after) concept to compare raw frames within lifted frames
    # Build square matrix
    dist_frames_glob = np.zeros((len(frames), len(frames)))

    for f_x, (lifted_x, frames_in_lifted_x) in enumerate(zip(frames, frames_list)):
        if isinstance(frames_in_lifted_x, np.int64):
            frames_in_lifted_x = [frames_in_lifted_x]
        for f_y, (lifted_y, frames_in_lifted_y) in enumerate(zip(frames, frames_list)):
            if isinstance(frames_in_lifted_y, np.int64):
                frames_in_lifted_y = [frames_in_lifted_y]

            # Left-right / previous-after concept
            if lifted_x < lifted_y:
                frame_in_lifted_x = frames_in_lifted_x[-1]  # Rightmost
                frame_in_lifted_y = frames_in_lifted_y[0]  # Leftmost
            elif lifted_x > lifted_y:
                frame_in_lifted_x = frames_in_lifted_x[0]  # Leftmost
                frame_in_lifted_y = frames_in_lifted_y[-1]  # Rightmost
            else:
                frame_in_lifted_x = frames_in_lifted_x[0]  # Same
                frame_in_lifted_y = frames_in_lifted_y[0]

            dist = abs(frame_in_lifted_x - frame_in_lifted_y)
            dist_frames_glob[f_x, f_y] = dist

    return dist_frames_glob


def get_same_frames_idx(frames, frames_list):
    # Function to check which are (truly) the frames that can't be fused
    # Previously we fused lifted frames and raw frames, maybe generating some unwanted separation of tracklets
    # Here we want to check whether an object is present (or not) in another cluster, rather than another lifted frame

    # Build square matrix
    same_frames_matrix = np.zeros((len(frames), len(frames)), dtype=bool)
    for f_x, (lifted_x, frames_in_lifted_x) in enumerate(zip(frames, frames_list)):
        for f_y, (lifted_y, frames_in_lifted_y) in enumerate(zip(frames, frames_list)):
            if (f_x == f_y) or (lifted_x[0] != lifted_y[0]):
                continue

            frames_in_lifted_x_np = np.array(frames_in_lifted_x)
            frames_in_lifted_y_np = np.array(frames_in_lifted_y)
            if isinstance(frames_in_lifted_x, np.int64):
                frames_in_lifted_x_np = np.array([int(frames_in_lifted_x)])
            if isinstance(frames_in_lifted_y, np.int64):
                frames_in_lifted_y_np = np.array([int(frames_in_lifted_y)])

            # Check frames really shared inside lifted frames
            share_any_frame = bool(set(frames_in_lifted_x_np) & set(frames_in_lifted_y_np))
            same_frames_matrix[f_x, f_y] = share_any_frame

    return same_frames_matrix


def bbox_dist_clust(x, frames, opt_motion=True):
    # Input: 2 lists or numpy arrays to compare
    # Output: distance matrix in pdist form for distance between bboxes (start with centroid)
    # Here we will do some weird things to compute IoU distance between tracklets.
    # Left-Right concept: a bbox comes from a frame, that can be left-right relative to other frames (before-after)
    # If the bbox is from a previous frame (or lifted frame), we will take the last bbox (bbox[-1]).
    # If the bbox is from a posterior frame (or lifted frame), we will take the first bbox (bbox[0]).
    # If the bbox is from the same frame (or lifted frame), we will take the same bbox (bbox[0] as standard).
    # This is done to enforce continuity between tracklets.
    # In FCG x=y, so we can use the index to check whether the frame is posterior or not.

    v = 0
    dist_bbox_glob = []
    for idx_x, (bboxes_1, frame_x) in enumerate(zip(x, frames)):
        dist_bbox_loc = []
        for idx_y, (bboxes_2, frame_y) in enumerate(zip(x, frames)):
            # Left-right / previous-after concept
            if frame_x < frame_y:
                # Test with very basic motion estimation (project left to right)
                if opt_motion:
                    v = motion_constant_velocity(bboxes_1)
                bbox_1 = bboxes_1[-1] + v  # Rightmost
                bbox_2 = bboxes_2[0]  # Leftmost
            elif frame_x > frame_y:
                if opt_motion:
                    v = motion_constant_velocity(bboxes_2)
                bbox_1 = bboxes_1[0]  # Leftmost
                bbox_2 = bboxes_2[-1] + v  # Rightmost
            else:
                bbox_1 = bboxes_1[0]  # Same bbox
                bbox_2 = bboxes_2[0]

            bbox_1_points = bbox_cx_cy_w_h_to_x1_y1_x2_y2(bbox_1)
            bbox_2_points = bbox_cx_cy_w_h_to_x1_y1_x2_y2(bbox_2)
            # Compute a scaled version of L2 distance between bboxes to capture
            # a) the displacement
            # b) the scale of the bbox
            # while the scale not be dominated by the bigger bounding box
            # Compute the distance between points of the bounding boxes (top-left vs. top-left, bot-right vs. bot-right)
            # relative to their size.
            mean_w = (bbox_1[2] + bbox_2[2]) / 2  # Scale by width and height of the bbox (not all displacements are equal)
            mean_h = (bbox_1[3] + bbox_2[3]) / 2
            dist_p1_x_rel = ((bbox_1_points[0] - bbox_2_points[0]) / mean_w) ** 2
            dist_p1_y_rel = ((bbox_1_points[1] - bbox_2_points[1]) / mean_h) ** 2
            dist_p2_x_rel = ((bbox_1_points[2] - bbox_2_points[2]) / mean_w) ** 2
            dist_p2_y_rel = ((bbox_1_points[3] - bbox_2_points[3]) / mean_h) ** 2
            # L2 distance
            l2_p1_rel = np.sqrt(dist_p1_x_rel + dist_p1_y_rel)
            l2_p2_rel = np.sqrt(dist_p2_x_rel + dist_p2_y_rel)
            l2_dist_rel = (l2_p1_rel + l2_p2_rel) / 2

            dist_bbox_loc.append(l2_dist_rel)

        dist_bbox_glob.append(dist_bbox_loc)

    return dist_bbox_glob


def bbox_dist(x, frames):
    # At 1st level we do normal IoU a list of bboxes against each other, after that we have a list of lists of bboxes
    # Here we can't do motion, as we haven't generated any tracklets (in form of clusters) yet

    dist_bbox_glob = []
    for idx_x, (bbox_1, frame_x) in enumerate(zip(x, frames)):
        dist_bbox_loc = []
        for idx_y, (bbox_2, frame_y) in enumerate(zip(x, frames)):
            bbox_1_points = bbox_cx_cy_w_h_to_x1_y1_x2_y2(bbox_1)
            bbox_2_points = bbox_cx_cy_w_h_to_x1_y1_x2_y2(bbox_2)
            # Compute a scaled version of L2 distance between bboxes to capture
            # a) the displacement
            # b) the scale of the bbox
            # while the scale not be dominated by the bigger bounding box
            # Compute the distance between points of the bounding boxes (top-left vs. top-left, bot-right vs. bot-right)
            # relative to their size.
            mean_w = (bbox_1[2] + bbox_2[2]) / 2  # Scale by width and height of the bbox (not all displacements are equal)
            mean_h = (bbox_1[3] + bbox_2[3]) / 2
            dist_p1_x_rel = ((bbox_1_points[0] - bbox_2_points[0]) / mean_w) ** 2
            dist_p1_y_rel = ((bbox_1_points[1] - bbox_2_points[1]) / mean_h) ** 2
            dist_p2_x_rel = ((bbox_1_points[2] - bbox_2_points[2]) / mean_w) ** 2
            dist_p2_y_rel = ((bbox_1_points[3] - bbox_2_points[3]) / mean_h) ** 2
            # L2 distance
            l2_p1_rel = np.sqrt(dist_p1_x_rel + dist_p1_y_rel)
            l2_p2_rel = np.sqrt(dist_p2_x_rel + dist_p2_y_rel)
            l2_dist_rel = (l2_p1_rel + l2_p2_rel) / 2

            dist_bbox_loc.append(l2_dist_rel)

        dist_bbox_glob.append(dist_bbox_loc)

    return dist_bbox_glob


def iou_dist_clust(x, frames, opt_motion=True):
    # Input: 2 lists or numpy arrays to compare
    # Output: distance matrix in pdist form for iou between bboxes
    # Here we will do some weird things to compute IoU distance between tracklets.
    # Left-Right concept: a bbox comes from a frame, that can be left-right relative to other frames (before-after)
    # If the bbox is from a previous frame (or lifted frame), we will take the last bbox (bbox[-1]).
    # If the bbox is from a posterior frame (or lifted frame), we will take the first bbox (bbox[0]).
    # If the bbox is from the same frame (or lifted frame), we will take the same bbox (bbox[0] as standard).
    # This is done to enforce continuity between tracklets.
    # In FCG x=y, so we can use the index to check whether the frame is posterior or not.

    v = 0
    dist_iou_glob = []
    for idx_x, (bboxes_x, frame_x) in enumerate(zip(x, frames)):
        dist_iou_loc = []
        for idx_y, (bboxes_y, frame_y) in enumerate(zip(x, frames)):
            # Left-right / previous-after concept
            if frame_x < frame_y:
                # Test with very basic motion estimation (project left to right)
                if opt_motion:
                    v = motion_constant_velocity(bboxes_x)
                bbox_x = bboxes_x[-1] + v  # Rightmost
                bbox_y = bboxes_y[0]  # Leftmost
            elif frame_x > frame_y:
                if opt_motion:
                    v = motion_constant_velocity(bboxes_y)
                bbox_x = bboxes_x[0]  # Leftmost
                bbox_y = bboxes_y[-1] + v  # Rightmost
            else:
                bbox_x = bboxes_x[0]  # Same bbox
                bbox_y = bboxes_y[0]

            bbox_x_points = bbox_cx_cy_w_h_to_x1_y1_x2_y2(bbox_x)
            bbox_y_points = bbox_cx_cy_w_h_to_x1_y1_x2_y2(bbox_y)
            iou = bb_intersection_over_union(bbox_x_points, bbox_y_points)
            dist_iou_loc.append(1 - iou)  # Return distances, not similarities (from 0 to 1)

        dist_iou_glob.append(dist_iou_loc)

    return dist_iou_glob


def iou_dist(x, frames):
    # At 1st level we do normal IoU a list of bboxes against each other, after that we have a list of lists of bboxes
    # Here we can't do motion, as we haven't generated any tracklets (in form of clusters) yet

    dist_iou_glob = []
    for idx_x, (bbox_x, frame_x) in enumerate(zip(x, frames)):
        dist_iou_loc = []
        bbox_x_points = bbox_cx_cy_w_h_to_x1_y1_x2_y2(bbox_x)
        for idx_y, (bbox_y, frame_y) in enumerate(zip(x, frames)):
            bbox_y_points = bbox_cx_cy_w_h_to_x1_y1_x2_y2(bbox_y)
            iou = bb_intersection_over_union(bbox_x_points, bbox_y_points)
            dist_iou_loc.append(1 - iou)  # Return distances, not similarities (from 0 to 1)

        dist_iou_glob.append(dist_iou_loc)

    return dist_iou_glob


def bb_intersection_over_union(boxA, boxB):
    # Source: https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
    # BBox form: x1, y1, x2, y2
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def bbox_x1_y1_x2_y2_to_cx_cy_w_h(bbox):
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]

    bbox_c_w = [cx, cy, w, h]
    return bbox_c_w


def bbox_cx_cy_w_h_to_x1_y1_x2_y2(bbox):
    x1 = bbox[0] - (bbox[2]/2)
    y1 = bbox[1] - (bbox[3]/2)
    x2 = bbox[0] + (bbox[2]/2)
    y2 = bbox[1] + (bbox[3]/2)

    bbox_x_y = [x1, y1, x2, y2]
    return bbox_x_y


def softmax(scores):
    scores_sum = np.sum(np.exp(scores))
    norm_scores = np.asarray([np.exp(s) / scores_sum for s in scores])

    return norm_scores
