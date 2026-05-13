from typing import Dict, List, Optional

import torch
import os
import math
from PIL import Image
import torchvision.transforms.functional as TF


def get_texture_config():
    keyword_order = ["diffuse", "normal", "roughness", "occlusion", "metallic", "specular", "displacement"]
    texture_keywords = {
        "diffuse": ["diffuse", "albedo", "color", "diff"],
        "normal": ["normal", "nor_gl"],
        "roughness": ["roughness", "rough"],
        "occlusion": ["occlusion", "ao", "ambient"],
        "metallic": ["metallic", "metalness"],
        "specular": ["specular"],
        "displacement": ["displacement", "disp", "height"],
    }
    texture_configs = {
        "diffuse":       {"expected_channels": 3, "color_mode": "RGB", "loss_weight": 1.0},
        "normal":        {"expected_channels": 3, "color_mode": "RGB", "loss_weight": 0.8},
        "roughness":     {"expected_channels": 1, "color_mode": "L",   "loss_weight": 0.3},
        "occlusion":     {"expected_channels": 1, "color_mode": "L",   "loss_weight": 0.3},
        "metallic":      {"expected_channels": 1, "color_mode": "L",   "loss_weight": 0.3},
        "specular":      {"expected_channels": 1, "color_mode": "L",   "loss_weight": 0.3},
        "displacement":  {"expected_channels": 1, "color_mode": "L",   "loss_weight": 0.3},
    }
    return keyword_order, texture_keywords, texture_configs


def _get_canonical_channel_slices():
    keyword_order, _, texture_configs = get_texture_config()
    slices = {}
    idx = 0
    for t in keyword_order:
        n = texture_configs[t]["expected_channels"]
        slices[t] = (idx, idx + n)
        idx += n
    return slices


CANONICAL_CHANNEL_SLICES = _get_canonical_channel_slices()
CANONICAL_NUM_CHANNELS = 11


class TextureDataset(torch.nn.Module):

    def __init__(self, data_dir: str, device: torch.device):
        super().__init__()

        self.device = device
        self.data_dir = data_dir

        self.keyword_order, self.texture_keywords, self.texture_configs = get_texture_config()
        self.channel_slices = {}
        self.available_textures = []

        self.textures = self._load_data()
        self.texture_height, self.texture_width, self.num_channels = self.textures.shape

        self.num_lods = int(min(math.log2(self.texture_height), math.log2(self.texture_width))) + 1
        self.lod_cache = self._generate_lod()

    @torch.no_grad()
    def forward(self, batch_index: torch.Tensor) -> torch.Tensor:
        ys = batch_index[:, 0]
        xs = batch_index[:, 1]
        lods = batch_index[:, 2]

        lod_scale = 2 ** lods
        scaled_xs = xs // lod_scale
        scaled_ys = ys // lod_scale

        return self.lod_cache[lods, scaled_ys, scaled_xs, :]

    def _load_data(self) -> torch.Tensor:
        filenames = os.listdir(self.data_dir)
        textures = {}
        for filename in filenames:
            if not filename.endswith(('.png', '.jpg', '.jpeg', '.tiff')):
                continue

            filepath = os.path.join(self.data_dir, filename)
            texture_type = self._identify_texture_type(filename)

            if texture_type not in self.texture_configs:
                raise ValueError(f"Unknown texture type: {texture_type}")

            cfg = self.texture_configs[texture_type]
            color_mode = cfg['color_mode']
            expected_channels = cfg['expected_channels']

            with Image.open(filepath) as image:
                image = image.convert(color_mode)
                tensor = TF.to_tensor(image)

                if texture_type != "normal":
                    tensor = torch.pow(tensor, 2.2)

                if tensor.shape[0] != expected_channels:
                    raise ValueError(
                        f"Expected {expected_channels} channels for '{texture_type}', "
                        f"got {tensor.shape[0]}"
                    )

            print(f"Loaded: type='{texture_type}', file='{filename}', shape={tensor.shape}")
            textures[texture_type] = tensor

        # Align all textures to the same resolution
        target_h, target_w = None, None
        for texture_type in self.keyword_order:
            if texture_type in textures:
                _, h, w = textures[texture_type].shape
                if target_h is None:
                    target_h, target_w = h, w
                elif h != target_h or w != target_w:
                    textures[texture_type] = TF.resize(
                        textures[texture_type], [target_h, target_w],
                        interpolation=TF.InterpolationMode.BICUBIC,
                        antialias=True,
                    )
                    textures[texture_type] = torch.clamp(textures[texture_type], 0.0, 1.0)

        # Concatenate in canonical order
        ordered = []
        current_index = 0
        self.channel_slices = {}
        self.available_textures = []
        for texture_type in self.keyword_order:
            if texture_type in textures and textures[texture_type] is not None:
                tex = textures[texture_type]
                ordered.append(tex)
                self.channel_slices[texture_type] = (current_index, current_index + tex.shape[0])
                self.available_textures.append(texture_type)
                current_index += tex.shape[0]

        return torch.cat(ordered, dim=0).permute(1, 2, 0).to(self.device)

    def _generate_lod(self) -> torch.Tensor:
        lod_cache = torch.zeros(
            [self.num_lods, self.texture_height, self.texture_width, self.num_channels]
        )
        textures = self.textures.cpu()

        for lod in range(self.num_lods):
            lod_h = self.texture_height // (2 ** lod)
            lod_w = self.texture_width // (2 ** lod)
            lod_texture = TF.resize(
                textures.permute(2, 0, 1), [lod_h, lod_w],
                interpolation=TF.InterpolationMode.BICUBIC,
                antialias=True,
            ).permute(1, 2, 0)
            lod_texture = torch.clamp(lod_texture, min=0., max=1.)
            lod_cache[lod, :lod_h, :lod_w, :] = lod_texture

        return lod_cache.to(self.device)

    def expand_to_canonical(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        out = torch.zeros((B, CANONICAL_NUM_CHANNELS), device=x.device, dtype=x.dtype)
        for tex_type in self.available_textures:
            ds_start, ds_end = self.channel_slices[tex_type]
            cn_start, cn_end = CANONICAL_CHANNEL_SLICES[tex_type]
            out[:, cn_start:cn_end] = x[:, ds_start:ds_end]
        return out

    def get_canonical_loss_weights(self, config_weights: Optional[Dict[str, float]] = None) -> List[float]:
        weights = [0.0] * CANONICAL_NUM_CHANNELS
        for tex_type in self.keyword_order:
            cn_start, cn_end = CANONICAL_CHANNEL_SLICES[tex_type]
            if tex_type not in self.available_textures:
                continue
            default_w = self.texture_configs[tex_type].get("loss_weight", 1.0)
            w = config_weights.get(tex_type, default_w) if config_weights else default_w
            for i in range(cn_start, cn_end):
                weights[i] = w
        return weights

    def _identify_texture_type(self, filename: str) -> str:
        filename_lower = filename.lower()
        for texture_type, keywords in self.texture_keywords.items():
            for keyword in keywords:
                if keyword in filename_lower:
                    return texture_type
        raise ValueError(f"Cannot identify texture type from filename: {filename}")
