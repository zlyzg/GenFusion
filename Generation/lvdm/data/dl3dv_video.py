import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import random
import json
import cv2


class ImageSequenceDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.video_list = []
        # for sub_dir in self.data_dir:
        #     print('sub_dir: ', sub_dir)
        #     self.video_list.extend([os.path.join(sub_dir, d) for d in os.listdir(sub_dir) if os.path.isdir(os.path.join(sub_dir, d))])

        # Read the JSON file
        json_path = "./filelist_long_with_video_depth.json"
        with open(json_path, "r") as f:
            self.video_list = json.load(f)

        # Ensure video_list is a list
        if not isinstance(self.video_list, list):
            raise ValueError("The JSON file should contain a list of video paths.")

        # Optionally, you can add full paths if the JSON contains relative paths
        # self.video_list = [os.path.join(self.data_dir, video_path) for video_path in self.video_list]
        self.transform = transform or transforms.ToTensor()
        print("len(self.video_list): ", len(self.video_list))
        self.crop_height = 512
        self.crop_width = 820
        self.input_height = 320
        self.input_width = 512
        self.num_frames = 16

    def __len__(self):
        return len(self.video_list)
        # return len(self.video_list)
        # return max(0, len(self.gt_image_files) - 15)  # Ensure at least 8 images are available

    def get_files_dir(self):
        self.artifact_dir = os.path.join(self.scene_dir, "renders_rgb")
        self.gt_dir = os.path.join(self.scene_dir, "gt_rgb")
        self.artifact_depth_dir = os.path.join(self.scene_dir, "renders_depth")
        self.gt_depth_dir = os.path.join(
            self.scene_dir.replace("_2dgs", ""), "gt_depth"
        )
        if not os.path.exists(self.gt_depth_dir):
            self.gt_depth_dir = os.path.join(self.scene_dir, "gt_depth")
        # self.gt_depth_dir = os.path.join(self.scene_dir, 'gt_depth')
        self.gt_image_files = sorted(
            [
                f
                for f in os.listdir(self.gt_dir)
                if f.endswith((".png", ".jpg", ".jpeg"))
            ]
        )
        self.gt_depth_files = sorted(
            [f for f in os.listdir(self.gt_depth_dir) if f.endswith(".npy")]
        )
        self.artifact_files = sorted(
            [
                f
                for f in os.listdir(self.artifact_dir)
                if f.endswith((".png", ".jpg", ".jpeg"))
            ]
        )
        self.artifact_depth_files = sorted(
            [f for f in os.listdir(self.artifact_depth_dir) if f.endswith(".npy")]
        )

    def read_rgb(self, img_path, start_x, start_y, crop=True):
        image = Image.open(img_path).convert("RGB")
        # Randomly select start x and y
        # Crop a [320, 512] region using start x and y
        if crop:
            image = transforms.functional.crop(
                image, start_y, start_x, self.crop_height, self.crop_width
            )
        image = self.transform(image)
        image = (image - 0.5) * 2  # Turn tensor to [-1, 1]
        return image

    def read_depth(self, depth_path, start_x, start_y):
        try:
            depth = np.load(depth_path)
        except Exception as e:
            print(f"Error loading depth file: {depth_path}")
            print(f"Error message: {str(e)}")
            raise  # Re-raise the exception after printing the information
        # print(depth_path, depth.shape)
        # Check if corresponding JSON file exists
        json_path = depth_path.replace(".npy", ".json")
        if os.path.exists(json_path):
            import json

            with open(json_path, "r") as f:
                depth_info = json.load(f)
            near = depth_info.get("min", 0.0)
            far = depth_info.get("max", 1.0)
            depth = depth * (far - near) + near
        if len(depth.shape) != 2:
            depth = depth.squeeze(0).squeeze(-1)
        depth = torch.from_numpy(depth).unsqueeze(0)  # Add channel dimension
        depth = depth[
            :, start_y : start_y + self.crop_height, start_x : start_x + self.crop_width
        ]
        return depth

    def read_video_depth(self, video_depth_dir, video_depth_paths, frame_idx):
        depth_sequence = []
        assert len(video_depth_paths) != 0, (
            f"{video_depth_dir}: video_depth_paths is empty"
        )
        for video_path in video_depth_paths:
            video_path = os.path.join(video_depth_dir, video_path)
            depths = np.load(video_path)
            array_name = depths.files
            assert len(array_name) == 1, "depth_merge should contain only one array"
            depths = depths[array_name[0]]
            depths = np.expand_dims(depths, axis=0).astype(np.float32)
            depth_sequence.append(depths)

        depth_sequence = np.concatenate(
            depth_sequence, axis=1
        )  # Concatenate along second dimension
        depth_sequence = torch.from_numpy(depth_sequence)  # Convert to torch tensor
        assert depth_sequence.shape[1] == len(self.gt_image_files), (
            f"{video_depth_dir}: depth_sequence and video_depth_paths should have the same length"
        )
        depth_sequence = depth_sequence[
            :, frame_idx : frame_idx + self.num_frames, :, :
        ]
        depth_sequence = depth_sequence[:, :, : self.crop_height, : self.crop_width]
        return depth_sequence

    def depth_normalize(self, depths):
        # Normalize depths using min-max normalization
        # depths_min = depths.min()
        # depths_max = depths.max()
        # normalized_depths = (depths - depths_min) / (depths_max - depths_min)
        # return normalized_depths
        # depths_min = depths.min()
        # return 1 - 1 / (depths - depths_min + 1)
        epsilon = 1e-10
        valid_mask = depths > 0
        disparity = torch.zeros_like(depths)
        disparity[valid_mask] = 1.0 / (depths[valid_mask] + epsilon)
        valid_disparities = torch.masked_select(disparity, valid_mask)
        if valid_disparities.numel() > 0:
            disp_min = valid_disparities.min()
            disp_max = valid_disparities.max()
            normalized_disparity = torch.zeros_like(disparity)
            normalized_disparity[valid_mask] = (disparity[valid_mask] - disp_min) / (
                disp_max - disp_min
            )
            # print("normalized_disparity:", normalized_disparity.max(), normalized_disparity.min())
        else:
            print("Warning: No valid depth values found")
            normalized_disparity = torch.zeros_like(disparity)

        return normalized_disparity

    def select_ref_frames(self, frame_idx, start_x, start_y):
        ref_frames = []
        frame_end = frame_idx + self.num_frames - 1
        ref_frame = torch.randint(frame_idx, frame_end + 1, (1,)).item()
        ref_img = self.read_rgb(
            os.path.join(self.gt_dir, self.gt_image_files[ref_frame]), 0, 0, crop=False
        )
        ref_frames = [ref_img]
        ref_frames = torch.stack(ref_frames).permute(1, 0, 2, 3)
        return ref_frames

    def __getitem__(self, index):
        # self.scene_dir = os.path.join(self.data_dir, self.video_list[index])
        self.scene_dir = self.video_list[
            index
        ]  # random.choice(self.video_list)#os.path.join(self.data_dir, random.choice(self.video_list))
        self.get_files_dir()
        gt_image_sequence = []
        depth_sequence = []
        artifact_image_sequence = []
        artifact_depth_sequence = []
        image = Image.open(os.path.join(self.gt_dir, self.gt_image_files[0])).convert(
            "RGB"
        )
        width, height = image.size
        start_x = 0  # torch.randint(0, max(1, width - self.crop_width), (1,)).item()
        start_y = 0  # torch.randint(0, max(1, height - self.crop_height), (1,)).item()
        frame_idx = torch.randint(
            0, max(1, len(self.gt_image_files) - self.num_frames + 1), (1,)
        ).item()
        ref_image_tensor = self.select_ref_frames(frame_idx, start_x, start_y)
        for i in range(frame_idx, frame_idx + self.num_frames):
            # Check if any of the indices are out of range
            if (
                i >= len(self.gt_image_files)
                or i >= len(self.artifact_files)
                or i >= len(self.gt_depth_files)
                or i >= len(self.artifact_depth_files)
            ):
                print(f"Warning: Index {i} out of range for scene: {self.scene_dir}")
                print(f"Length of gt_image_files: {len(self.gt_image_files)}")
                print(f"Length of artifact_files: {len(self.artifact_files)}")
                print(f"Length of gt_depth_files: {len(self.gt_depth_files)}")
                print(
                    f"Length of artifact_depth_files: {len(self.artifact_depth_files)}"
                )

            gt_img_path = os.path.join(self.gt_dir, self.gt_image_files[i])
            gt_image = self.read_rgb(gt_img_path, start_x, start_y)
            gt_image_sequence.append(gt_image)

            artifact_img_path = os.path.join(self.artifact_dir, self.artifact_files[i])
            if i >= len(self.artifact_files):
                print(f"Error: artifact_files index {i} out of range")
                print(f"artifact_dir: {self.artifact_dir}")
                print(f"Number of artifact files: {len(self.artifact_files)}")
                raise IndexError(f"artifact_files index {i} is out of range")
            artifact_image = self.read_rgb(artifact_img_path, start_x, start_y)
            artifact_image_sequence.append(artifact_image)
            if i > len(self.gt_depth_files):
                print(
                    f"Warning: gt_depth_files index {i} out of range for {self.scene_dir}"
                )
            # depth_path = os.path.join(self.gt_depth_dir, self.gt_depth_files[i])
            # depth = self.read_depth(depth_path, start_x, start_y)
            # depth_sequence.append(depth)

            artifact_depth_path = os.path.join(
                self.artifact_depth_dir, self.artifact_depth_files[i]
            )
            artifact_depth = self.read_depth(artifact_depth_path, start_x, start_y)
            artifact_depth_sequence.append(artifact_depth)
        video_depth_dir = self.scene_dir.replace("DL3DV_gs_output", "DL3DV-compressed")
        video_depth_paths = sorted(
            [f for f in os.listdir(video_depth_dir) if f.endswith(".npz")]
        )
        gt_depth_tensor = self.read_video_depth(
            video_depth_dir, video_depth_paths, frame_idx
        )
        # Stack the 8 images into a single tensor and permute dimensions
        gt_image_tensor = torch.stack(gt_image_sequence).permute(
            1, 0, 2, 3
        )  # [c, t, h, w]
        artifact_image_tensor = torch.stack(artifact_image_sequence).permute(
            1, 0, 2, 3
        )  # [c, t, h, w]
        ######### Depth #########
        # gt_depth_tensor = torch.stack(depth_sequence).permute(1, 0, 2, 3)  # [c, t, h, w]
        # gt_depth_tensor = self.depth_normalize(
        # gt_depth_tensor)  # Attention: we need disparity instead of depth as input
        gt_depth_tensor = (gt_depth_tensor - 0.5) * 2
        artifact_depth_tensor = torch.stack(artifact_depth_sequence).permute(
            1, 0, 2, 3
        )  # [c, t, h, w]
        artifact_depth_tensor = self.depth_normalize(artifact_depth_tensor)
        artifact_depth_tensor = (artifact_depth_tensor - 0.5) * 2
        # Normalize gt_image_tensor and artifact_image_tensor to [3, 16, 320, 512]
        # if self.input_height != self.crop_height or self.input_width != self.crop_width:
        gt_image_tensor = torch.nn.functional.interpolate(
            gt_image_tensor.unsqueeze(0),
            size=(self.num_frames, self.input_height, self.input_width),
            mode="trilinear",
            align_corners=False,
        ).squeeze(0)
        artifact_image_tensor = torch.nn.functional.interpolate(
            artifact_image_tensor.unsqueeze(0),
            size=(self.num_frames, self.input_height, self.input_width),
            mode="trilinear",
            align_corners=False,
        ).squeeze(0)

        # Normalize gt_depth_tensor and artifact_depth_tensor to [1, 16, 320, 512]
        gt_depth_tensor = torch.nn.functional.interpolate(
            gt_depth_tensor.unsqueeze(0),
            size=(self.num_frames, self.input_height, self.input_width),
            mode="nearest",
        ).squeeze(0)
        artifact_depth_tensor = torch.nn.functional.interpolate(
            artifact_depth_tensor.unsqueeze(0),
            size=(self.num_frames, self.input_height, self.input_width),
            mode="nearest",
        ).squeeze(0)

        # Normalize ref_image_tensor to [3, 2, 320, 512]
        ref_image_tensor = torch.nn.functional.interpolate(
            ref_image_tensor.unsqueeze(0),
            size=(1, self.input_height, self.input_width),
            mode="trilinear",
            align_corners=False,
        ).squeeze(0)

        data = {
            "caption": self.scene_dir,
            "ref_rgb": ref_image_tensor,
            "gt_rgb": gt_image_tensor,
            "gt_depth": gt_depth_tensor,
            "artifact_rgb": artifact_image_tensor,
            "artifact_depth": artifact_depth_tensor,
            "path": "./data",
            "fps": 30,
            "frame_stride": 1,
        }
        return data
