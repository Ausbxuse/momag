import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def cache_frames(video):
    frames = []
    success, frame = video.read()
    while success:
        frames.append(frame)
        success, frame = video.read()
    return frames


def gen_colors(n_colors=10):
    cmap = plt.get_cmap("tab10", n_colors)
    colors = (cmap(np.arange(n_colors))[:, :3] * 255).astype(np.uint8)
    return colors


def write_traj(
    frames, trajectories, feat_idx=0, labels=None, output_file="./out/output_video.avi"
):
    frames = frames.permute(0, 2, 3, 1)
    frames = frames.detach().cpu().numpy()

    # frames = np.transpose(frames, (1, 2, 0)).astype(np.uint8)

    if labels is None:
        labels = [0 for _ in range(trajectories.shape[0])]
    if not isinstance(trajectories, np.ndarray):
        trajectories = trajectories.numpy()
    height, width, _ = frames[0].shape
    fps = 30

    fourcc = cv2.VideoWriter_fourcc(*"XVID")  # Use XVID or appropriate codec
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    n_features = trajectories.shape[0]
    color_map = gen_colors()
    colors = np.array([[color_map[label]] for label in labels])

    for k in tqdm(range(len(frames)), desc="Writing video with trajectories"):
        frame = frames[k].copy()

        for n in range(n_features):
            x, y = trajectories[n, k]

            if not np.isnan(x) and not np.isnan(y) and x >= 0 and y >= 0:
                if n == feat_idx:
                    cv2.circle(frame, (int(x), int(y)), 2, [255, 0, 255], -1)
                else:
                    cv2.circle(frame, (int(x), int(y)), 2, colors[n].tolist()[0], -1)

        # frame = draw_features_w_corr_mat(frame, trajectories[:, k, :], corr_mat)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)

    out.release()
    print(f"Video saved to {output_file}")


def calc_affine_motion(features_a, features_b):
    affine_matrix, _ = cv2.estimateAffinePartial2D(
        features_a,
        features_b,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
    )
    return affine_matrix


def apply_affine_warp(affine_matrix, frame_k):

    frame_k = frame_k.permute(1, 2, 0).cpu().numpy()

    h, w = frame_k.shape[:2]
    aligned_frame = cv2.warpAffine(
        frame_k,
        affine_matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    return aligned_frame


def draw_features(frame, features):
    frame = frame.permute(1, 2, 0).cpu().numpy()

    for point in features:
        x, y = point.astype(int)
        cv2.circle(frame, (x, y), 2, (0, 255, 0), thickness=2)


def draw_features_w_corr_mat(frame, features, corr_mat):
    for idx, point in enumerate(features):
        x, y = point.astype(int)
        color = np.array([0, 255, 0]) * corr_mat[0, idx]
        if idx == 0:
            color = (0, 0, 255)
        cv2.circle(frame, (x, y), 2, color, thickness=2)
    return frame
