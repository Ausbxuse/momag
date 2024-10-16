from torchvision.io import write_jpeg
from torchvision.utils import flow_to_image
import os
import numpy as np
from PIL import Image
import cProfile
import pstats
import io
import subprocess
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torchvision.io import read_video, write_video
from torchvision.models.optical_flow import raft_large
from torchvision.models.optical_flow import Raft_Large_Weights
import torchvision.transforms.functional as F
import torchvision.transforms as T
from stablize import stablize_video, get_all_features, get_traj_lk, get_traj
from util import write_traj
from sam2.sam2_video_predictor import SAM2VideoPredictor
import torch.nn.functional as NF
from diffusers import StableDiffusionInpaintPipeline


def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

    # frame_stride = 30
    # for frame_id in range(0, 100, frame_stride):
    #     plt.figure(figsize=(6, 4))
    #     plt.title(f"frame {frame_id}")
    #     for out_obj_id, out_mask in video_segments[frame_id].items():
    #         show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
    #     plt.savefig(f"out_{frame_id}.png")


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )


def make_video_dir(video_file, name):
    print("Spliting frames into jpegs")
    result = subprocess.run(
        [
            "ffmpeg",
            "-i",
            video_file,
            "-q:v",
            "2",
            "-start_number",
            "0",
            f"{name}/%05d.jpg",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    print("finished spliting frames: ", result.stdout)


# ffmpeg -i <your_video>.mp4 -q:v 2 -start_number 0 <output_dir>/'%05d.jpg'
def add_initial_masks(video_dir, predictor, obj_id_map):
    inference_state = predictor.init_state(video_path=video_dir)

    starting_frame_idx = 0  # the frame index we interact with

    for obj_id, (points, labels) in obj_id_map.items():
        points = np.array([[200, 210], [250, 400]], dtype=np.float32)
        labels = np.array([1, 1], np.int32)

        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=starting_frame_idx,
            obj_id=obj_id,
            points=points,
            labels=labels,
        )

        plt.figure(figsize=(9, 6))
        plt.title(f"frame {starting_frame_idx}")
        # plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
        show_points(points, labels, plt.gca())
        show_mask(
            (out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0]
        )
        plt.savefig(f"initial_masks_of_obj_{obj_id}.png")
    return inference_state


def propagate_masks(predictor, inference_state):
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
        inference_state
    ):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    return video_segments


def preprocess(img1_batch, img2_batch):
    weights = Raft_Large_Weights.DEFAULT
    transforms = weights.transforms()
    img1_batch = F.resize(img1_batch, size=[H, W], antialias=False)
    img2_batch = F.resize(img2_batch, size=[H, W], antialias=False)
    return transforms(img1_batch, img2_batch)


def get_flows(frames, raft):
    chunk_size = 13
    all_flows = []
    num_frames = len(frames)

    # ffmpeg -f image2 -framerate 30 -i predicted_flow_%d.jpg -loop -1 flow.gif
    image_idx = 0

    for idx in tqdm(
        range(0, num_frames - 1, chunk_size), desc="Getting flows from frames"
    ):
        start_idx = idx
        end_idx = min(idx + chunk_size, num_frames - 1)

        img1_batch = frames[start_idx:end_idx]
        img2_batch = frames[start_idx + 1 : end_idx + 1]

        img1_batch, img2_batch = preprocess(img1_batch, img2_batch)

        with torch.no_grad():
            flow = raft(img1_batch.to("cuda"), img2_batch.to("cuda"))[-1].cpu()

        flow_imgs = flow_to_image(flow)
        # flow = flow.cpu()
        for imgs in flow_imgs:
            image_idx += 1
            write_jpeg(imgs, f"./out/flow/predicted_flow_{image_idx}.jpg")

        all_flows.append(flow)

    all_flows = torch.cat(all_flows, dim=0)
    return all_flows


def inpaint_texture(frame, mask_value=0.0):
    inpainted_frame = frame.clone()  # [1, c, h, w]
    frame_tensor = frame.squeeze(0)

    frame_np = frame_tensor.cpu().numpy()
    frame_np = np.transpose(frame_np, (1, 2, 0))
    frame_np = (frame_np * 255).astype(np.uint8)

    image = Image.fromarray(frame_np)

    mask = (inpainted_frame == mask_value).all(dim=1, keepdim=True)  # [1, 1, h, w]
    mask_np = mask.cpu().numpy()
    mask_np = (mask_np * 255).astype(np.uint8)
    mask_image = Image.fromarray(np.squeeze(mask_np))

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting", torch_dtype=torch.float16
    ).to("cuda")

    result = pipe(prompt="fill the background", image=image, mask_image=mask_image)
    inpainted_image = result.images[0]
    inpainted_np = np.array(inpainted_image)

    inpainted_np = inpainted_np.astype(np.float32) / 255.0
    inpainted_tensor = torch.from_numpy(inpainted_np)
    inpainted_tensor = inpainted_tensor.permute(2, 0, 1)

    if frame_tensor.dim() == 4:
        inpainted_tensor = inpainted_tensor.unsqueeze(0)
    inpainted_tensor = inpainted_tensor.type(frame_tensor.dtype)
    inpainted_tensor = inpainted_tensor.to(frame_tensor.device)

    return inpainted_frame


def magnify_motion(frames, flows, video_segments, id, mag_ratio=40):
    n_frames, channels, h, w = frames.shape
    frames_out = frames.clone()

    # create a normalized grid
    grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    grid_x = grid_x.to(frames.device).float()
    grid_y = grid_y.to(frames.device).float()
    grid_x_norm = 2.0 * grid_x / (w - 1) - 1.0
    grid_y_norm = 2.0 * grid_y / (h - 1) - 1.0
    base_grid = torch.stack((grid_x_norm, grid_y_norm), 2)  # [h, w, 2]

    for k in range(1, n_frames):
        flow = flows[k - 1]  # [2, h, w]
        mask = video_segments[k - 1][id]
        mask = torch.from_numpy(mask).to(flow.device).float()  # [h, w]

        frame = frames[k - 1].unsqueeze(0).float()  # [1, c, h, w]

        # print("mask", mask)
        magnified_flow = flow * mag_ratio

        flow_u = magnified_flow[0, :, :] * 2 / (w - 1)
        flow_v = magnified_flow[1, :, :] * 2 / (h - 1)
        flow_norm = torch.stack((flow_u, flow_v), 2)  # [h, w, 2]
        new_grid = base_grid + flow_norm
        new_grid = new_grid.unsqueeze(0).float()  # [1, h, w, 2]

        warped_frame = NF.grid_sample(
            frame,
            new_grid,
            mode="bicubic",
            padding_mode="border",
            align_corners=True,
        )

        mask = mask.unsqueeze(0)

        # print("mask shape", mask.shape)
        warped_mask = NF.grid_sample(
            mask,
            new_grid,
            mode="nearest",
            padding_mode="zeros",
            align_corners=True,
        )

        warped_mask = (warped_mask >= 0.5).float()
        # if k == 1:
        #     frames[k] = inpaint_texture(
        #         frames[k] * (1 - warped_mask) * (1 - mask)
        #     )
        # else:
        frames_out[k] = warped_frame * warped_mask + frames_out[k] * (1 - warped_mask)

    return frames_out


def resize_video(frames, h, w):
    resize_transform = T.Resize((h, w))
    resized_frames = []
    for frame in frames:
        frame = resize_transform(frame)
        resized_frames.append(frame)

    frames = torch.stack(resized_frames)
    return frames


if __name__ == "__main__":
    ################ PROFILING ##################
    # pr = cProfile.Profile()
    # pr.enable()
    ################ PROFILING ##################

    INPUT_DIR = "./in/"
    INPUT_FILE = "swing.mp4"
    obj_id_map = {}
    obj_id_map[1] = ([[200, 210], [250, 400]], [1, 1])
    magnification_rate = 400

    name, ext = os.path.splitext(INPUT_FILE)
    output_file = f"{name}-magnified.mp4"
    video_dir = INPUT_DIR + name

    if not os.path.isdir(video_dir):
        make_video_dir(INPUT_DIR + INPUT_FILE, video_dir)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if device.type == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    print(f"using device: {device}")

    raft = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device)
    raft = raft.eval()
    predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-large")

    frames, _, _ = read_video(INPUT_DIR + INPUT_FILE, output_format="TCHW")
    _, _, h, w = frames.shape
    H = h // 8 * 8
    W = w // 8 * 8
    # H = 960
    # W = 520
    frames = resize_video(frames, H, W)
    print(f"resized from {h, w} to {H, W}")

    stable_video = stablize_video(frames, INPUT_FILE)

    frames_stable, _, _ = read_video(stable_video, output_format="TCHW")
    frames_stable = resize_video(frames_stable, H, W)

    print("stable_video shape is ", frames_stable.shape)

    inference_state = add_initial_masks(video_dir, predictor, obj_id_map)
    video_segments = propagate_masks(predictor, inference_state)

    flows = get_flows(frames_stable, raft)
    print("flows shape is ", flows.shape)
    # print("flows is ", flows)
    print("Magnifying")
    frames_mag = magnify_motion(
        frames_stable, flows, video_segments, 1, magnification_rate
    )  # magnifies obj 1
    print("Finished magnifying")

    frames_mag = frames_mag.permute(0, 2, 3, 1)  # [T, H, W, C]

    write_video(output_file, frames_mag, fps=30)
    print(f"Wrote {output_file} with shape {frames_mag.shape}")

    ################ PROFILING ##################
    # pr.disable()
    # s = io.StringIO()
    # sortby = pstats.SortKey.TIME
    # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    # ps.print_stats()
    # print("\n".join(s.getvalue().splitlines()[:10]))
    ################ PROFILING ##################
