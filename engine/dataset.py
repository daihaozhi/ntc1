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

    def __init__(self, data_dir: str, device: torch.device, diffuse_color_space: str = "linear"):
        super().__init__()

        if diffuse_color_space not in {"linear", "srgb"}:
            raise ValueError("diffuse_color_space must be 'linear' or 'srgb'")

        self.device = device
        self.data_dir = data_dir
        self.diffuse_color_space = diffuse_color_space

        self.keyword_order, self.texture_keywords, self.texture_configs = get_texture_config()
        self.channel_slices = {}
        self.available_textures = []

        self.textures = self._load_data()
        self.texture_height, self.texture_width, self.num_channels = self.textures.shape

        self.num_lods = int(min(math.log2(self.texture_height), math.log2(self.texture_width))) + 1

        # The model output is built from the maps actually present in the
        # dataset. Keep the historical material order, then append data maps.
        # For the current data this becomes diffuse(3)+normal(3)+roughness+
        # occlusion+displacement = 9 channels.
        self.model_texture_order = [
            "diffuse", "metallic", "normal", "roughness",
            "occlusion", "displacement", "specular",
        ]
        self.model_channel_indices = []
        self.model_channel_names = []
        self.model_channel_types = []
        self.model_texture_slices = {}
        for tex_type in self.model_texture_order:
            if tex_type not in self.available_textures:
                continue
            cn_start, cn_end = CANONICAL_CHANNEL_SLICES[tex_type]
            out_start = len(self.model_channel_indices)
            self.model_channel_indices.extend(range(cn_start, cn_end))
            self.model_texture_slices[tex_type] = (out_start, len(self.model_channel_indices))
            for channel in range(cn_start, cn_end):
                suffix = "rgb"[channel - cn_start] if cn_end - cn_start == 3 else "value"
                self.model_channel_names.append(f"{tex_type}.{suffix}")
                self.model_channel_types.append(tex_type)

        self.model_output_dim = len(self.model_channel_indices)
        self.lod_cache = self._generate_lod()

    @torch.no_grad()
    def forward(self, batch_index: torch.Tensor) -> torch.Tensor:
        return self.sample_discrete_lod(batch_index)

    @torch.no_grad()
    def sample_discrete_lod(self, batch_index: torch.Tensor) -> torch.Tensor:
        """Sample one texel from a discrete mip level.

        batch_index columns are full-resolution y, full-resolution x, and integer lod.
        This keeps the original nearest-texel training target path available.
        """
        ys = batch_index[:, 0]
        xs = batch_index[:, 1]
        lods = batch_index[:, 2]

        lod_scale = 2 ** lods
        scaled_xs = xs // lod_scale
        scaled_ys = ys // lod_scale

        return self.lod_cache[lods, scaled_ys, scaled_xs, :]

    @torch.no_grad()
    def sample_trilinear_lod(self, uv: torch.Tensor, lod: torch.Tensor) -> torch.Tensor:
        """Sample the generated mip chain with bilinear-in-mip and linear-between-mips filtering.

        uv is normalized [0, 1) texture space and lod is a continuous mip value.
        This produces the same kind of target a hardware texture unit would return
        for trilinear filtering, but from the generated training mip chain.
        """
        if lod.ndim == 2:
            lod = lod.squeeze(1)

        uv = torch.remainder(uv, 1.0)
        lod = torch.clamp(lod, 0.0, float(self.num_lods - 1))
        lod0 = torch.floor(lod).to(torch.long)
        lod1 = torch.clamp(lod0 + 1, max=self.num_lods - 1)
        t = (lod - lod0.float()).unsqueeze(1)

        c0 = self._sample_mip_bilinear(uv, lod0)
        c1 = self._sample_mip_bilinear(uv, lod1)
        return c0 * (1.0 - t) + c1 * t

    def _sample_mip_bilinear(self, uv: torch.Tensor, lods: torch.Tensor) -> torch.Tensor:
        out = torch.empty((uv.shape[0], self.num_channels), device=self.device, dtype=self.textures.dtype)
        for lod in torch.unique(lods):
            mask = lods == lod
            if not mask.any():
                continue

            lod_int = int(lod.item())
            lod_h = max(self.texture_height >> lod_int, 1)
            lod_w = max(self.texture_width >> lod_int, 1)
            mip = self.lod_cache[lod_int, :lod_h, :lod_w, :]

            # uv denotes normalized texture coordinates at texel centers.
            # Convert to texel-space so a center sample maps to the texel
            # itself (the usual half-texel convention).
            p = uv[mask] * torch.tensor([lod_w, lod_h], device=self.device, dtype=uv.dtype) - 0.5
            p0 = torch.floor(p).to(torch.long)
            f = p - p0.float()

            x0 = torch.remainder(p0[:, 0], lod_w)
            y0 = torch.remainder(p0[:, 1], lod_h)
            x1 = torch.remainder(x0 + 1, lod_w)
            y1 = torch.remainder(y0 + 1, lod_h)

            c00 = mip[y0, x0, :]
            c10 = mip[y0, x1, :]
            c01 = mip[y1, x0, :]
            c11 = mip[y1, x1, :]

            fx = f[:, [0]]
            fy = f[:, [1]]
            c0 = c00 * (1.0 - fx) + c10 * fx
            c1 = c01 * (1.0 - fx) + c11 * fx
            out[mask] = c0 * (1.0 - fy) + c1 * fy

        return out

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

                # Diffuse is the only color texture here.  Data textures
                # (normal, roughness, AO, displacement) must retain their
                # stored numeric values; applying gamma to them changes the
                # material parameter itself.
                if texture_type == "diffuse" and self.diffuse_color_space == "linear":
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
