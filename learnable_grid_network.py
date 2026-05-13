import json
import math
import torch
import torch.nn as nn
import tinycudann as tcnn


class LearnableGridNetwork(nn.Module):
    DEFAULT_GRID_RESOLUTIONS = {
        0: [256, 128],
        1: [64, 32],
        2: [16, 8],
        3: [4, 2],
    }

    def __init__(
        self,
        grid_configs: dict[int, list[dict]] | None = None,
        grid_config_path: str | None = None,
        texture_resolution: int = 1024,
        default_save_bits: int = 32,
        default_quantize_bits: int = 8,
        output_dim: int = 11,
        hidden_dim: int = 64,
        num_hidden_layers: int = 2,
        quantize: bool = True,
        max_iter: int = 10000,
        qat_noise_schedule: str = "cosine",
        qat_noise_mult_start: float = 1.0,
        qat_noise_mult_end: float = 0.25,
        qat_noise_warmup_frac: float = 0.0,
        use_tiled_encoding: bool = False,
        tile_size: int = 8,
        n_frequencies: int = 8,
    ):
        super().__init__()

        # Tiled positional encoding configuration
        self.use_tiled_encoding = use_tiled_encoding
        self.tile_size = tile_size
        self.n_frequencies = n_frequencies
        
        if self.use_tiled_encoding:
            print(f"Using Tiled Positional Encoding with tile_size={tile_size}x{tile_size}, n_frequencies={n_frequencies}")
        else:
            print("Using standard TriangleWave Positional Encoding")

        if grid_config_path is not None:
            with open(grid_config_path, "r", encoding="utf-8") as f:
                raw_config = json.load(f)
            tex_key = str(texture_resolution)
            if tex_key not in raw_config:
                raise ValueError(
                    f"Texture resolution {texture_resolution} not found in {grid_config_path}. "
                    f"Available: {list(raw_config.keys())}"
                )
            tex_cfg = raw_config[tex_key]
            # Parse per-feature-grid configs with mip ranges
            parsed = {}
            self.level_mip_ranges = []
            self.num_mip_levels = 0
            for fg_idx, fg_cfg in enumerate(tex_cfg["feature_grids"]):
                mip_lo, mip_hi = fg_cfg["mip_range"]
                self.level_mip_ranges.append([mip_lo, mip_hi])
                self.num_mip_levels = max(self.num_mip_levels, mip_hi)
                level_configs = []
                for res in fg_cfg["resolutions"]:
                    level_configs.append({
                        "resolution": res,
                        "save_bits": fg_cfg.get("save_bits", default_save_bits),
                        "quantize_bits": fg_cfg.get("quantize_bits", default_quantize_bits),
                    })
                parsed[fg_idx] = level_configs
            grid_configs = parsed
        else:
            self.level_mip_ranges = None
            self.num_mip_levels = 0

        self.grid_configs = self._build_grid_configs(
            grid_configs=grid_configs,
            default_save_bits=default_save_bits,
            default_quantize_bits=default_quantize_bits,
        )
        
        # Automatically identify the higher-resolution grid in each level
        self.high_res_grid_indices = {}
        for level, grid_cfg_list in self.grid_configs.items():
            if len(grid_cfg_list) >= 2:
                # Find the grid with the highest resolution
                max_res_idx = 0
                max_res = grid_cfg_list[0]["resolution"]
                for i, cfg in enumerate(grid_cfg_list):
                    if cfg["resolution"] > max_res:
                        max_res = cfg["resolution"]
                        max_res_idx = i
                self.high_res_grid_indices[level] = max_res_idx
            elif len(grid_cfg_list) == 1:
                # Only one grid, treat it as high-res
                self.high_res_grid_indices[level] = 0
        
        self.output_dim = output_dim
        self.quantize = quantize
        self.max_iter = max(1, int(max_iter))
        self.current_iter = 0
        self._qat_noise_schedule = str(qat_noise_schedule).strip().lower()
        self._qat_noise_mult_start = float(qat_noise_mult_start)
        self._qat_noise_mult_end = float(qat_noise_mult_end)
        self._qat_noise_warmup_frac = float(qat_noise_warmup_frac)

        # Fallback when no config-based mip info (e.g., grid_configs passed directly)
        if self.level_mip_ranges is None:
            if self.num_mip_levels == 0:
                self.num_mip_levels = 11  # default for 1024 texture
            num_fg_levels = len(self.grid_configs)
            self.level_mip_ranges = []
            for l in range(num_fg_levels):
                start = int(round(l * self.num_mip_levels / num_fg_levels))
                end = int(round((l + 1) * self.num_mip_levels / num_fg_levels))
                self.level_mip_ranges.append([start, end])

        self.grids = nn.ModuleDict()
        self.grid_save_bits = nn.ModuleDict()
        self.grid_quantize_bits = nn.ModuleDict()
        self.grid_feature_dims = nn.ModuleDict()
        total_feature_dim = 0
        self.level_feature_dims = []  # per-level feature dimension

        for level, grid_cfg_list in self.grid_configs.items():
            level_grids = nn.ModuleList()
            level_save_bits = nn.ParameterList()
            level_quantize_bits = nn.ParameterList()
            level_feature_dims = nn.ParameterList()
            level_dim = 0

            for grid_cfg in grid_cfg_list:
                resolution = int(grid_cfg["resolution"])
                save_bits = int(grid_cfg["save_bits"])
                quantize_bits = int(grid_cfg["quantize_bits"])
                feature_dim = save_bits // quantize_bits
                
                # Determine if this is the high-res grid in this level
                is_high_res = (len(grid_cfg_list) >= 2 and 
                              grid_cfg_list.index(grid_cfg) == self.high_res_grid_indices.get(level, 0))
                
                # High-res grid outputs 4x features (4 corners), low-res output 1x
                if is_high_res:
                    total_feature_dim += feature_dim * 4
                    level_dim += feature_dim * 4
                else:
                    total_feature_dim += feature_dim
                    level_dim += feature_dim

                # Use Nearest for high-res grid (we'll manually sample corners), Linear for low-res
                interp_mode = "Nearest" if is_high_res else "Linear"

                level_grids.append(
                    tcnn.Encoding(
                        n_input_dims=2,
                        encoding_config={
                            "otype": "Grid",
                            "type": "Dense",
                            "n_levels": 1,
                            "n_features_per_level": feature_dim,
                            "base_resolution": resolution,
                            "per_level_scale": 1.0,
                            "interpolation": interp_mode,
                        },
                    )
                )
                level_save_bits.append(nn.Parameter(torch.tensor(float(save_bits)), requires_grad=False))
                level_quantize_bits.append(nn.Parameter(torch.tensor(float(quantize_bits)), requires_grad=False))
                level_feature_dims.append(nn.Parameter(torch.tensor(float(feature_dim)), requires_grad=False))

            self.grids[str(level)] = level_grids
            self.grid_save_bits[str(level)] = level_save_bits
            self.grid_quantize_bits[str(level)] = level_quantize_bits
            self.grid_feature_dims[str(level)] = level_feature_dims
            self.level_feature_dims.append(level_dim)

        # New network input:
        #   single level features + position encoding + lod value
        single_level_dim = self.level_feature_dims[0]
        n_input_dims = single_level_dim + (n_frequencies * 2) + 1

        self.network = tcnn.Network(
            n_input_dims=n_input_dims,
            n_output_dims=output_dim,
            network_config={
                "otype": "CutlassMLP",
                "activation": "LeakyReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim,
                "n_hidden_layers": num_hidden_layers,
            },
        )

    def _build_grid_configs(
        self,
        grid_configs: dict[int, list[dict]] | None,
        default_save_bits: int,
        default_quantize_bits: int,
    ) -> dict[int, list[dict]]:
        if grid_configs is None:
            return {
                level: [
                    {
                        "resolution": res,
                        "save_bits": default_save_bits,
                        "quantize_bits": default_quantize_bits,
                    }
                    for res in resolutions
                ]
                for level, resolutions in self.DEFAULT_GRID_RESOLUTIONS.items()
            }

        normalized_configs = {}
        for level_key, cfg_list in grid_configs.items():
            level = int(level_key)
            normalized_configs[level] = []
            for grid_cfg in cfg_list:
                resolution = int(grid_cfg["resolution"])
                save_bits = int(grid_cfg.get("save_bits", default_save_bits))
                quantize_bits = int(grid_cfg.get("quantize_bits", default_quantize_bits))
                if resolution <= 0:
                    raise ValueError(f"resolution must be positive, got {resolution}")
                if save_bits <= 0:
                    raise ValueError(f"save_bits must be positive, got {save_bits}")
                if quantize_bits <= 0:
                    raise ValueError(f"quantize_bits must be positive, got {quantize_bits}")
                if save_bits % quantize_bits != 0:
                    raise ValueError(f"save_bits must be divisible by quantize_bits, got {save_bits} and {quantize_bits}")

                normalized_configs[level].append(
                    {
                        "resolution": resolution,
                        "save_bits": save_bits,
                        "quantize_bits": quantize_bits,
                    }
                )

        return normalized_configs

    def _triangle_wave_encoding(self, x: torch.Tensor) -> torch.Tensor:
        """Compute triangle wave positional encoding.
        
        Triangle wave function: tri(x) = 2 * |x - floor(x + 0.5)|
        This creates a periodic triangular waveform in [0, 1].
        
        For each frequency i, we compute:
          - tri(2^i * x)
        
        Args:
            x: Input coordinates [B, 2] in range [0, 1)
            
        Returns:
            Encoded features [B, n_frequencies * 2]
        """
        batch_size = x.shape[0]
        encoded_features = []
        
        for i in range(self.n_frequencies):
            # Compute frequency scale: 2^i
            freq_scale = 2.0 ** i
            
            # Apply frequency scaling
            scaled_x = x * freq_scale  # [B, 2]
            
            # Triangle wave: tri(u) = 2 * |u - floor(u + 0.5)|
            # This creates a triangular wave oscillating between 0 and 1
            triangle = 2.0 * torch.abs(scaled_x - torch.floor(scaled_x + 0.5))
            
            encoded_features.append(triangle)
        
        # Concatenate all frequencies: [B, n_frequencies * 2]
        return torch.cat(encoded_features, dim=1)

    def _compute_tiled_local_coords(self, uvs: torch.Tensor) -> torch.Tensor:
        """Compute local coordinates within each tile for tiled positional encoding.
        
        This maps global UV coordinates to local tile coordinates:
        - Divides the texture space into tile_size x tile_size blocks
        - Each point is mapped to its relative position within its tile
        - Different tiles with same relative positions get identical encodings
        
        Args:
            uvs: UV coordinates [B, 2] in range [0, 1)
            
        Returns:
            Local coordinates within tiles [B, 2] in range [0, 1)
        """
        # Scale UV by tile_size and take fractional part
        # This effectively wraps the coordinate space every tile_size units
        local_uvs = torch.remainder(uvs * self.tile_size, 1.0)
        
        return local_uvs

    def _compute_positional_encoding(self, uv: torch.Tensor) -> torch.Tensor:
        """Compute positional encoding using triangle wave.
        
        If tiled encoding is enabled, uses local tile coordinates.
        Otherwise, uses global UV coordinates.
        
        Args:
            uv: UV coordinates [B, 2] in range [0, 1)
            
        Returns:
            Positional encoding features [B, n_frequencies * 2]
        """
        if self.use_tiled_encoding:
            # Use local coordinates within tiles
            local_uvs = self._compute_tiled_local_coords(uv)
            return self._triangle_wave_encoding(local_uvs)
        else:
            # Use global UV coordinates
            return self._triangle_wave_encoding(uv)

    def _qat_noise_multiplier(self) -> float:
        if self._qat_noise_schedule == "none":
            return 1.0
        T = max(1, self.max_iter - 1)
        t = min(max(int(self.current_iter), 0), T)
        w = int(self._qat_noise_warmup_frac * T)
        if t <= w:
            return self._qat_noise_mult_start
        p = (t - w) / max(1, T - w)
        c = 0.5 * (1.0 + math.cos(math.pi * p))
        return self._qat_noise_mult_end + (self._qat_noise_mult_start - self._qat_noise_mult_end) * c

    def _simulate_quantize(self, features: torch.Tensor, quantize_bits: int) -> torch.Tensor:
        N_k = 2 ** quantize_bits
        Q_k = 1.0 / N_k

        noise_range = 0.5 * Q_k
        mult = self._qat_noise_multiplier()
        noise = (torch.rand_like(features) * 2 - 1) * noise_range * mult
        features_noisy = features + noise

        quantized = torch.round(features_noisy / Q_k) * Q_k

        # STE: forward uses quantized, backward uses original features
        return features + (quantized - features).detach()

    def _is_high_res_grid(self, level: int, grid_index: int) -> bool:
        """Check if a specific grid in a level is the high-resolution grid."""
        return self.high_res_grid_indices.get(level, 0) == grid_index

    def _sample_grid_corners(self, grid: tcnn.Encoding, uv: torch.Tensor, resolution: int) -> torch.Tensor:
        """Sample 4 corner points from a dense grid without interpolation.
        
        For each UV coordinate, this returns the features at the 4 surrounding grid corners:
        (floor_u, floor_v), (ceil_u, floor_v), (floor_u, ceil_v), (ceil_u, ceil_v)
        
        Args:
            grid: tcnn Grid encoding object
            uv: UV coordinates [B, 2] in range [0, 1)
            resolution: Grid resolution
            
        Returns:
            Features at 4 corners [B, 4 * feature_dim]
        """
        batch_size = uv.shape[0]
        
        # Convert UV to grid coordinates
        grid_uv = uv * resolution  # [B, 2]
        
        # Get floor and ceil indices
        floor_u = torch.floor(grid_uv[:, 0]).long()  # [B]
        floor_v = torch.floor(grid_uv[:, 1]).long()  # [B]
        ceil_u = (floor_u + 1) % resolution  # Wrap around
        ceil_v = (floor_v + 1) % resolution  # Wrap around
        
        # Create 4 corner coordinates (normalized back to [0, 1))
        corners_uv = torch.stack([
            torch.stack([floor_u.float() / resolution, floor_v.float() / resolution], dim=1),  # bottom-left
            torch.stack([ceil_u.float() / resolution, floor_v.float() / resolution], dim=1),   # bottom-right
            torch.stack([floor_u.float() / resolution, ceil_v.float() / resolution], dim=1),   # top-left
            torch.stack([ceil_u.float() / resolution, ceil_v.float() / resolution], dim=1),    # top-right
        ], dim=1)  # [B, 4, 2]
        
        # Reshape for batch processing: [B*4, 2]
        corners_uv_flat = corners_uv.reshape(-1, 2)
        
        # Sample all corners at once using nearest interpolation
        # Since we're sampling at exact grid points, nearest will give us the exact values
        corner_features = grid(corners_uv_flat)  # [B*4, feature_dim]
        
        # Reshape back to [B, 4, feature_dim]
        feature_dim = corner_features.shape[1]
        corner_features = corner_features.reshape(batch_size, 4, feature_dim)
        
        # Flatten to [B, 4 * feature_dim]
        return corner_features.reshape(batch_size, -1)

    def sample_features(
        self,
        uv: torch.Tensor,
        level: int | None = None,
        grid_index: int | None = None,
    ) -> torch.Tensor:
        uv = torch.remainder(uv, 1.0)

        if level is not None:
            if level not in self.grid_configs:
                raise ValueError(f"level must be one of {list(self.grid_configs.keys())}, got {level}")

            level_grids = self.grids[str(level)]
            level_qbits = self.grid_quantize_bits[str(level)]
            level_fdims = self.grid_feature_dims[str(level)]
            
            if grid_index is not None:
                grid_cfg = self.grid_configs[level][grid_index]
                resolution = grid_cfg["resolution"]
                
                if self._is_high_res_grid(level, grid_index):
                    # High-res grid: sample 4 corners
                    feat = self._sample_grid_corners(level_grids[grid_index], uv, resolution)
                else:
                    # Low-res grid: use bilinear interpolation via tcnn
                    feat = level_grids[grid_index](uv)
                
                if self.quantize and self.training:
                    feat = self._simulate_quantize(feat, int(level_qbits[grid_index].item()))
                return feat

            feats = []
            for i, grid in enumerate(level_grids):
                grid_cfg = self.grid_configs[level][i]
                resolution = grid_cfg["resolution"]
                
                if self._is_high_res_grid(level, i):
                    # High-res grid: sample 4 corners
                    feat = self._sample_grid_corners(grid, uv, resolution)
                else:
                    # Low-res grid: use bilinear interpolation via tcnn
                    feat = grid(uv)
                
                if self.quantize and self.training:
                    feat = self._simulate_quantize(feat, int(level_qbits[i].item()))
                feats.append(feat)
            return torch.cat(feats, dim=1)

        if grid_index is not None:
            raise ValueError("grid_index can only be used together with level.")

        features = []
        for level_key in self.grids:
            level_int = int(level_key)
            level_grids = self.grids[level_key]
            level_qbits = self.grid_quantize_bits[level_key]
            
            for i, grid in enumerate(level_grids):
                grid_cfg = self.grid_configs[level_int][i]
                resolution = grid_cfg["resolution"]
                
                if self._is_high_res_grid(level_int, i):
                    # High-res grid: sample 4 corners
                    feat = self._sample_grid_corners(grid, uv, resolution)
                else:
                    # Low-res grid: use bilinear interpolation via tcnn
                    feat = grid(uv)
                
                if self.quantize and self.training:
                    feat = self._simulate_quantize(feat, int(level_qbits[i].item()))
                features.append(feat)
        return torch.cat(features, dim=1)

    def _compute_level_index(self, mip: torch.Tensor) -> torch.Tensor:
        """Map continuous mip level to a single feature grid level index.

        Uses bucketize with the per-level mip range boundaries.
        E.g. level_mip_ranges = [[0,4], [4,6], [6,8], [8,11]]
          → boundaries = [0, 4, 6, 8, 11]
          → mip in [0,4) → level 0, [4,6) → level 1, [6,8) → level 2, [8,11) → level 3

        Args:
            mip: Mip values [B, 1], range [0, num_mip_levels)

        Returns:
            Level indices [B, 1], each in [0, num_feature_grid_levels)
        """
        boundaries = torch.tensor(
            [r[0] for r in self.level_mip_ranges] + [self.num_mip_levels],
            dtype=torch.float32, device=mip.device,
        )
        idx = torch.bucketize(mip, boundaries) - 1
        return idx.clamp(min=0, max=len(self.grid_configs) - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        uv = torch.remainder(x[:, 0:2], 1.0)
        lod = x[:, [2]]  # [B, 1]

        mip = lod * (self.num_mip_levels - 1)  # [B, 1]
        level_idx = self._compute_level_index(mip)  # [B, 1]

        pos_encoding = self._compute_positional_encoding(uv)  # [B, n_frequencies * 2]

        # Sample features from the selected level
        features = self.sample_features(uv, level=level_idx[0, 0].item())

        combined = torch.cat([pos_encoding, features, lod], dim=1)

        return self.network(combined)

    def get_grid_params(self, level: int, grid_index: int) -> torch.nn.Parameter:
        if level not in self.grid_configs:
            raise ValueError(f"level must be one of {list(self.grid_configs.keys())}, got {level}")

        grid = self.grids[str(level)][grid_index]
        for name, param in grid.named_parameters():
            if name == "params":
                return param
        return next(grid.parameters())

    def get_grid_config(self, level: int, grid_index: int) -> dict[str, int]:
        if level not in self.grid_configs:
            raise ValueError(f"level must be one of {list(self.grid_configs.keys())}, got {level}")

        return dict(self.grid_configs[level][grid_index])


if __name__ == "__main__":
    model = LearnableGridNetwork(
        grid_config_path="grid_config.json",
        texture_resolution=1024,
    ).cuda()
    x = torch.rand(8, 3, device="cuda")  # (u, v, lod)
    rgb = model(x)

    print(f"input (u, v, lod) shape: {tuple(x.shape)}")
    print(f"output rgb shape: {tuple(rgb.shape)}")
    print(f"level_mip_ranges: {model.level_mip_ranges}")
    print(f"num_mip_levels: {model.num_mip_levels}")

    # Test different LoD values
    for lod_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
        x_test = torch.tensor([[0.5, 0.5, lod_val]], device="cuda")
        out = model(x_test)
        mip = lod_val * (model.num_mip_levels - 1)
        print(f"  LoD={lod_val:.2f} → mip≈{mip:.1f} → rgb={out[0].cpu().detach().numpy().round(3)}")

    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"trainable parameters: {trainable_params}")
    