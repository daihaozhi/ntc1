# Tiled Positional Encoding 使用说明

## 概述

在 `learnable_grid_network.py` 中实现了 **Tiled Positional Encoding**，使用三角波函数进行位置编码。

## 核心特性

### 1. 三角波函数 (Triangle Wave)

三角波函数定义：
```
tri(x) = 2 * |x - floor(x + 0.5)|
```

特性：
- 周期性：周期为 1
- 值域：[0, 1]
- 分段线性，梯度稳定

对于每个频率 i (i = 0, 1, ..., n_frequencies-1)：
```
encoding_i = tri(2^i * x)
```

### 2. Tiled 分块策略

将纹理空间分成 `tile_size × tile_size` 的块：

```python
local_uv = (uv * tile_size) % 1.0
```

**效果**：
- Tile (0,0) 中相对位置 (0.5, 0.5) 的点
- Tile (1,1) 中相对位置 (0.5, 0.5) 的点
- 这两个点会有**完全相同的位置编码**

### 3. 参数说明

```python
model = LearnableGridNetwork(
    grid_config_path="grid_config.json",
    
    # Tiled Positional Encoding 配置
    use_tiled_encoding=True,   # 是否启用 tiled encoding
    tile_size=8,               # tile 大小 (8x8 像素)
    n_frequencies=8,           # 三角波频率数量
    
    # ... 其他参数
)
```

## 使用示例

### 示例 1: 启用 Tiled Encoding

```python
import torch
from learnable_grid_network import LearnableGridNetwork

# 创建模型（启用 tiled encoding）
model = LearnableGridNetwork(
    grid_config_path="grid_config.json",
    use_tiled_encoding=True,
    tile_size=8,
    n_frequencies=8,
).cuda()

# 前向传播
uv = torch.rand(16, 2, device="cuda")
rgb = model(uv)

print(f"Input: {uv.shape}")   # [16, 2]
print(f"Output: {rgb.shape}") # [16, 3]
```

### 示例 2: 标准 Encoding（无分块）

```python
# 创建模型（使用标准全局 encoding）
model = LearnableGridNetwork(
    grid_config_path="grid_config.json",
    use_tiled_encoding=False,  # 禁用 tiled
    n_frequencies=8,
).cuda()

uv = torch.rand(16, 2, device="cuda")
rgb = model(uv)
```

### 示例 3: 验证位置复用

```python
model = LearnableGridNetwork(
    grid_config_path="grid_config.json",
    use_tiled_encoding=True,
    tile_size=8,
    n_frequencies=8,
).cuda()

# 两个不同 tile 中相同相对位置的点
test_uvs = torch.tensor([
    [0.01, 0.01],    # Tile (0,0), local pos ≈ (0.08, 0.08)
    [0.135, 0.135],  # Tile (1,1), local pos ≈ (0.08, 0.08)
], device="cuda")

# 计算局部坐标
local_uvs = model._compute_tiled_local_coords(test_uvs)
print(f"UV[0]: {test_uvs[0]}, Local: {local_uvs[0]}")
print(f"UV[1]: {test_uvs[1]}, Local: {local_uvs[1]}")
print(f"Local coords match: {torch.allclose(local_uvs[0], local_uvs[1])}")
# 输出: True

# 计算位置编码
pos_enc = model._compute_positional_encoding(test_uvs)
print(f"Encodings match: {torch.allclose(pos_enc[0], pos_enc[1])}")
# 输出: True
```

## 工作原理

### 1. 位置编码流程

```
输入 UV 坐标
    ↓
[如果启用 tiled]
    ↓
计算局部坐标: local_uv = (uv * tile_size) % 1.0
    ↓
三角波编码:
  for i in range(n_frequencies):
    freq_scale = 2^i
    scaled = local_uv * freq_scale
    triangle = 2 * |scaled - floor(scaled + 0.5)|
    ↓
拼接所有频率 → [B, n_frequencies * 2]
    ↓
与 Grid 特征拼接
    ↓
MLP 网络 → RGB 输出
```

### 2. 特征维度

假设配置：
- `n_frequencies = 8`
- Grid 总特征维度 = 80（来自之前的 4 角点采样）

则网络输入维度：
```
n_input_dims = (n_frequencies * 2) + total_grid_features
             = (8 * 2) + 80
             = 16 + 80
             = 96
```

### 3. Tile 边界处理

使用 `torch.remainder` 实现无缝 wrap：

```python
# UV = 0.125 (正好在 tile 边界)
local_uv = (0.125 * 8) % 1.0 = 1.0 % 1.0 = 0.0

# UV = 0.126 (刚过边界)
local_uv = (0.126 * 8) % 1.0 = 1.008 % 1.0 = 0.008
```

这确保了 tile 边界的连续性。

## 优势

### 1. 参数效率
- 不同 tile 共享相同的编码模式
- 减少模型需要学习的全局变化

### 2. 局部性
- 编码专注于 tile 内的局部结构
- 更好地捕捉高频细节

### 3. 可扩展性
- 支持任意大小的纹理
- tile 大小可调整以平衡局部性和全局信息

### 4. 周期性
- 三角波的周期性天然支持纹理平铺
- 避免边界不连续问题

## 测试

运行测试脚本验证实现：

```bash
python test_tiled_positional_encoding.py
```

测试内容包括：
1. 标准 encoding vs tiled encoding 对比
2. Tile 边界行为验证
3. 完整前向传播测试
4. 网络输入维度验证
5. 三角波函数性质测试（周期性、值域）

## 配置建议

### Tile Size 选择

- **tile_size = 4**: 更小的 tile，更强的局部性，适合高频细节
- **tile_size = 8**: 默认值，平衡局部性和全局信息
- **tile_size = 16**: 更大的 tile，更多的全局上下文

### Frequency 数量

- **n_frequencies = 4**: 较少频率，平滑编码
- **n_frequencies = 8**: 默认值，良好的频率覆盖
- **n_frequencies = 12**: 更多频率，捕捉更精细的变化

## 注意事项

1. **网络输入维度自动调整**：添加位置编码后，网络输入维度会自动增加 `n_frequencies * 2`

2. **训练兼容性**：tiled encoding 不影响梯度流动，可以正常训练

3. **与 Grid 特征的协同**：位置编码提供全局/局部位置信息，Grid 特征提供纹理细节，两者互补

4. **内存开销**：位置编码增加的内存开销很小（仅 `n_frequencies * 2` 维）
