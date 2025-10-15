# env_parallel_repair.py
import os
import argparse
import imageio
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from einops import repeat, reduce
import multiprocessing as mp
from multiprocessing import Manager, Queue
from types import SimpleNamespace
import queue

from Reconstruction.args import config_args
from Reconstruction.extrapolation.extrapolator import Extrapolator

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7" 

# --------------------- helpers (same as your originals) --------------------- #
def load_video(video_path):
    reader = imageio.get_reader(video_path)
    frames = []
    for frame in reader:
        pil_img = Image.fromarray(frame)
        frames.append(pil_img)
    reader.close()
    return frames

def save_video(frames, save_path, fps, quality=9, ffmpeg_params=None):
    writer = imageio.get_writer(save_path, fps=fps, quality=quality, ffmpeg_params=ffmpeg_params)
    for frame in frames:
        frame = np.array(frame)
        writer.append_data(frame)
    writer.close()

def preprocess_image(image, torch_dtype=None, device=None, pattern="B C H W", min_value=-1, max_value=1):
    image = torch.Tensor(np.array(image, dtype=np.float32))
    image = image.to(dtype=torch_dtype, device=device)
    image = image * ((max_value - min_value) / 255) + min_value
    image = repeat(image, f"H W C -> {pattern}", **({"B": 1} if "B" in pattern else {}))
    return image

def preprocess_video(video, torch_dtype=None, device=None, pattern="B C T H W", min_value=-1, max_value=1):
    video = [preprocess_image(image, torch_dtype=torch_dtype, device=device, min_value=min_value, max_value=max_value) for image in video]
    video = torch.stack(video, dim=pattern.index("T") // 2)
    return video

def save_tensor_video(tensor, save_path="output.mp4", fps=24):
    assert tensor.ndim == 4, f"Expected (C, T, H, W), got {tensor.shape}"
    C, T, H, W = tensor.shape
    assert C == 3, "Only 3-channel (RGB) tensors supported"
    video = tensor.detach().cpu().numpy()
    video = np.transpose(video, (1, 2, 3, 0))
    if video.min() < 0:
        video = (video + 1) / 2.0
    video = np.clip(video, 0, 1)
    video = (video * 255).astype(np.uint8)
    writer = imageio.get_writer(save_path, fps=fps, codec="libx264")
    for frame in video:
        writer.append_data(frame)
    writer.close()
    print(f"Saved video to {save_path}")

def vae_output_to_video(vae_output, pattern="B C T H W", min_value=-1, max_value=1):
    if pattern != "T H W C":
        vae_output = reduce(vae_output, f"{pattern} -> T H W C", reduction="mean")
    video = [vae_output_to_image(image, pattern="H W C", min_value=min_value, max_value=max_value) for image in vae_output]
    return video

def vae_output_to_image(vae_output, pattern="B C H W", min_value=-1, max_value=1):
    if pattern != "H W C":
        vae_output = reduce(vae_output, f"{pattern} -> H W C", reduction="mean")
    image = ((vae_output - min_value) * (255 / (max_value - min_value))).clip(0, 255)
    image = image.to(device="cpu", dtype=torch.uint8)
    image = Image.fromarray(image.numpy())
    return image

def load_scene(scene_path, torch_dtype=None, device=None):
    rgb_path = os.path.join(scene_path, "color")
    depth_path = os.path.join(scene_path, "depth")
    video_ids = os.listdir(rgb_path)
    scene_data = []
    for video_id in video_ids:
        rgb_video_path = os.path.join(rgb_path, video_id)
        depth_video_path = os.path.join(depth_path, video_id)
        rgb_frames = load_video(rgb_video_path)
        depth_frames = load_video(depth_video_path)
        rgb_video = preprocess_video(rgb_frames, torch_dtype=torch_dtype, device=device).squeeze()
        depth_video = preprocess_video(depth_frames, torch_dtype=torch_dtype, device=device).squeeze()

        target_h, target_w = 512, 960
        C, T, H, W = rgb_video.shape
        rgb_video = F.interpolate(rgb_video, size=(target_h, target_w), mode="bilinear", align_corners=False)

        C, T, H, W = depth_video.shape
        depth_video = F.interpolate(depth_video, size=(target_h, target_w), mode="bilinear", align_corners=False)

        scene_data.append((video_id, rgb_video, depth_video))
    return scene_data

def process_scene(extrapolator, scene_path, torch_dtype, device):
    scene_data = load_scene(scene_path, torch_dtype=torch_dtype, device=device)
    save_dir = os.path.join(scene_path, "GenFusion")
    os.makedirs(save_dir, exist_ok=True)
    for video_id, rgb_video, depth_video in scene_data:
        pred_save_path = os.path.join(save_dir, video_id)
        if os.path.exists(pred_save_path):
            continue
        with torch.inference_mode():
            _, _, pred_tensor, _ = extrapolator.repair(rgb_video[:], depth_video[:1], rgb_video[:, :1])
        pred_video = vae_output_to_video(pred_tensor, pattern="C T H W")
        save_video(pred_video, pred_save_path, fps=18)
# --------------------------------------------------------------------------- #

def list_scenes(dataset_path):
    return sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])

def parse_gpu_ids_from_env():
    """
    Returns a list of *physical* GPU IDs to use.
    - If CUDA_VISIBLE_DEVICES is already set in the parent shell, we honor that ordering.
    - Otherwise, we enumerate 0..torch.cuda.device_count()-1.
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cvd:
        # e.g. "0,2,4,5"
        return [int(x) for x in cvd.split(",")]
    return list(range(torch.cuda.device_count()))

def build_extrapolator(args_ns, device, dtype):
    # If your Extrapolator reads args.device/dtype, set them here:
    setattr(args_ns, "device", device)
    setattr(args_ns, "dtype", dtype)
    torch.backends.cudnn.benchmark = True
    return Extrapolator(args_ns)

def worker_process(phys_gpu_id, args_dict, work_q: Queue):
    # Pin this process to ONE GPU by setting CUDA_VISIBLE_DEVICES to a single ID.
    # After this, the only visible GPU is logical cuda:0 for this process.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(phys_gpu_id)

    # Now resolve device; only one GPU is visible, so use cuda:0
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16  # switch to bfloat16 if your model prefers it

    args = SimpleNamespace(**args_dict)
    extrapolator = build_extrapolator(args, device, dtype)

    while True:
        try:
            scene_id = work_q.get_nowait()
        except queue.Empty:
            break
        scene_path = os.path.join(args.dataset_path, scene_id)
        try:
            print(f"[GPU {phys_gpu_id}] -> {scene_id}")
            process_scene(extrapolator, scene_path, dtype, device)
        except Exception as e:
            print(f"[GPU {phys_gpu_id}] Error on {scene_id}: {e}")

def main():
    # CLI wrapper to merge your config_args() with dataset_path
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dataset_path", type=str,
                        default="/home/liyuanz/workspace/DiffSynth-Studio/logs/benchmark_splat")
    known, _ = parser.parse_known_args()

    base_args = config_args()
    setattr(base_args, "dataset_path", known.dataset_path)

    # Scenes to process
    scenes = list_scenes(base_args.dataset_path)
    print(f"Found {len(scenes)} scenes under: {base_args.dataset_path}")

    # GPU pool from env or all visible
    gpu_ids = parse_gpu_ids_from_env()
    if not gpu_ids:
        raise RuntimeError("No GPUs found. Set CUDA_VISIBLE_DEVICES or ensure CUDA is available.")
    print(f"Using GPUs (physical IDs): {gpu_ids}")

    # Shared work queue (dynamic load balancing)
    mp.set_start_method("spawn", force=True)
    with Manager() as mgr:
        work_q = mgr.Queue()
        for s in scenes:
            work_q.put(s)

        args_dict = vars(base_args)  # make picklable
        procs = []
        for gid in gpu_ids:
            p = mp.Process(target=worker_process, args=(gid, args_dict, work_q), daemon=False)
            p.start()
            procs.append(p)

        for p in procs:
            p.join()

    print("All processes completed.")

if __name__ == "__main__":
    main()
