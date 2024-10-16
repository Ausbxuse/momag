import os
from skimage import exposure
import torch
from torchvision.io import read_video, write_video
import cv2
import numpy as np
from util import (
    write_traj,
    calc_affine_motion,
    apply_affine_warp,
)


def get_all_features(frame):
    # mask = mask.squeeze().numpy()

    # mask = (mask * 255).astype(np.uint8)
    # print(mask[0].shape)

    frame = frame.permute(1, 2, 0).cpu().numpy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    feature_sets = []  # contains features from different scales
    for scale in [1.0, 0.5, 0.25]:
        resized_gray = cv2.resize(
            gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR
        )
        # resized_mask = cv2.resize(
        #     mask[0], None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR
        # )
        features = cv2.goodFeaturesToTrack(
            resized_gray,
            maxCorners=500,
            qualityLevel=0.01,
            minDistance=7,
            blockSize=7,
            # mask=resized_mask,
            useHarrisDetector=True,
            k=0.1,
        )
        if features is not None:
            features /= scale
            feature_sets.append(features)

    feature_sets = np.vstack(feature_sets).astype(np.float32)
    print(f"Feature set shape is {feature_sets.shape} from extracting features")
    return np.squeeze(feature_sets, axis=1)


# default starting on the first frame
def get_traj(feat, flows):
    n_feat = feat.shape[0]
    n_frames = flows.shape[0] + 1

    feat = torch.from_numpy(feat).to(torch.float32)

    traj = torch.zeros((n_feat, n_frames, 2), dtype=torch.float32, device=feat.device)
    traj[:, 0, :] = feat
    print("feature reference shape is ", feat.shape)
    print("flows shape is ", flows.shape)

    feat_curr = feat
    for k in range(1, n_frames):
        x_coords = traj[:, k - 1, 0].long()
        y_coords = traj[:, k - 1, 1].long()

        traj[:, k, 0] = feat_curr[:, 0] + flows[k - 1, 0, y_coords, x_coords]  # x
        traj[:, k, 1] = feat_curr[:, 1] + flows[k - 1, 1, y_coords, x_coords]  # y
        feat_curr = traj[:, k, :]
    print("### [get_traj] got trajectory shape ", traj.shape)

    return traj


def get_traj_lk(frames, feat_ref):

    frame_ref = frames[0]
    frame_ref = frame_ref.permute(1, 2, 0).cpu().numpy()
    frame_ref = cv2.cvtColor(frame_ref, cv2.COLOR_BGR2GRAY)

    n_features = feat_ref.shape[0]
    n_frames = len(frames)
    trajectories = np.zeros((n_features, n_frames, 2), dtype=np.float32)

    trajectories[:, 0, :] = feat_ref

    good_points_mask = np.ones(len(feat_ref), dtype=bool)
    for k in range(1, n_frames):
        frame_k = frames[k].permute(1, 2, 0).cpu().numpy()
        frame_k = cv2.cvtColor(frame_k, cv2.COLOR_BGR2GRAY)

        tracked_points, status, _ = cv2.calcOpticalFlowPyrLK(
            frame_ref,
            frame_k,
            feat_ref,
            None,
            winSize=(7, 7),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )
        good_points_mask &= status.ravel().astype(bool)

        trajectories[:, k, :] = tracked_points

    trajectories = trajectories[good_points_mask]

    return trajectories


def calc_stable_prob(k, trajectories):
    # trajectories of shape (n, k, 2) where n is number of features, k frames, 2 dimensions

    features_ref = trajectories[:, 0, :].astype(np.float32)
    features_k = trajectories[:, k, :].astype(np.float32)

    v_nk = features_k - features_ref  # shape (n, 2)

    a_k = calc_affine_motion(features_ref, features_k)

    n_features = features_ref.shape[0]
    features_coord = np.hstack([features_ref, np.ones((n_features, 1))])

    predicted_motion = (a_k @ features_coord.T).T - features_ref
    # print("Predicted motion", predicted_motion)
    # print("v_nk", v_nk)

    diff = predicted_motion - v_nk
    norm_sq = np.linalg.norm(diff, axis=1) ** 2

    sigma_k = np.mean(norm_sq)
    pr_k = np.exp(
        -norm_sq / (2 * sigma_k**2)
    )  # calculates for all n features. shape (n)

    return pr_k


def get_stable_traj(trajectories):
    n_features, n_frames, _ = trajectories.shape
    alpha = 0.3

    pr_nk = np.ones(n_features)  # for storing final stable prob of all features
    max_pr_nk = np.zeros(n_features)
    for k in range(1, n_frames):
        pr_k = calc_stable_prob(
            k, trajectories
        )  # stable prob of all features in frame k
        pr_nk *= pr_k
        max_pr_nk = np.maximum(max_pr_nk, pr_nk)  # the maximum

    # print("pr_nk is ", pr_nk)
    # print("max_pr_nk is ", max_pr_nk)
    threshold = (alpha**n_frames) * max_pr_nk
    # threshold = alpha
    # print("Thresh hold is ", threshold)

    stable_indices = np.where(pr_nk > threshold)[0]
    traj_stable = trajectories[stable_indices]

    print(f"There are {traj_stable.shape[0]}/{n_features} stable features")

    return traj_stable


def equalize_histogram(frame_k, frame_ref):
    frame = exposure.match_histograms(frame_k, frame_ref, channel_axis=-1)
    return frame


def stablize_video(frames, filename, flows=None):  # frames are pytorch tensors
    n_frames = len(frames)
    name, ext = os.path.splitext(filename)
    out_file = "./out/{name}_stable{ext}".format(name=name, ext=ext)

    frame_ref = frames[0]
    feat_bg = get_all_features(frame_ref)

    traj_bg = get_traj_lk(frames, feat_bg)  # NOTE: consider using raft
    # traj_bg = get_traj(feat_bg, flows).numpy()

    traj_bg = get_stable_traj(traj_bg)

    write_traj(frames, traj_bg)
    print("traj_bg shape: ", traj_bg.shape)

    frames_out = [frame_ref.permute(1, 2, 0)]

    features_ref = traj_bg[:, 0, :]
    for k in range(1, n_frames):
        features_k = traj_bg[:, k, :]

        a_k = calc_affine_motion(
            features_k.astype(np.float32), features_ref.astype(np.float32)
        )
        frame_aligned = apply_affine_warp(a_k, frames[k])  # converted to cv object
        frame_ref_cv = frame_ref.permute(1, 2, 0).cpu().numpy()
        frame_aligned = equalize_histogram(frame_aligned, frame_ref_cv)

        frames_out.append(torch.from_numpy(frame_aligned).float())

    frames_out = torch.stack(frames_out)
    # print(frames_out.shape)

    write_video(out_file, frames_out, fps=30)
    print("## [stablize_video] Saved stablized video to", out_file)
    return out_file
