# 4角点采样策略说明

## 修改概述

修改了 `learnable_grid_network.py`，实现了对每个level中**分辨率较大的grid**使用4角点采样策略，而不是双线性插值。

## 采样策略

### 之前的行为
- 所有grid都使用双线性插值（Linear interpolation）
- 每个查询点输出1个特征值

### 现在的行为
- **每个level中分辨率较大的grid**：输出4个角点的特征值（不插值）
- **每个level中分辨率较小的grid**：保持双线性插值

## 具体配置示例

根据 `grid_config.json`：

```
Level 0:
  - Grid 0 (256x256): 4角点采样 → 输出 4×4=16 维特征
  - Grid 1 (128x128): 双线性插值 → 输出 4 维特征

Level 1:
  - Grid 0 (64x64):   4角点采样 → 输出 4×4=16 维特征
  - Grid 1 (32x32):   双线性插值 → 输出 4 维特征

Level 2:
  - Grid 0 (16x16):   4角点采样 → 输出 4×4=16 维特征
  - Grid 1 (8x8):     双线性插值 → 输出 4 维特征

Level 3:
  - Grid 0 (4x4):     4角点采样 → 输出 4×4=16 维特征
  - Grid 1 (2x2):     双线性插值 → 输出 4 维特征

总特征维度: 16+4 + 16+4 + 16+4 + 16+4 = 80 维
```

## 4角点采样原理

对于UV坐标 `(u, v)` 在分辨率 `R` 的网格上：

1. 计算网格坐标：`(u*R, v*R)`
2. 找到4个相邻角点：
   - 左下角：`(floor(u*R), floor(v*R))`
   - 右下角：`(ceil(u*R), floor(v*R))`
   - 左上角：`(floor(u*R), ceil(v*R))`
   - 右上角：`(ceil(u*R), ceil(v*R))`
3. 使用wrap-around处理边界：`(index + 1) % R`
4. 采样这4个角点的特征值并拼接

## 实现细节

### 关键方法

1. **`high_res_grid_indices`**: 自动识别每个level中分辨率最大的grid索引
   ```python
   self.high_res_grid_indices = {
       0: 0,  # Level 0的第0个grid (256) 是高分辨率
       1: 0,  # Level 1的第0个grid (64) 是高分辨率
       2: 0,  # Level 2的第0个grid (16) 是高分辨率
       3: 0,  # Level 3的第0个grid (4) 是高分辨率
   }
   ```

2. **`_is_high_res_grid(level, grid_index)`**: 判断指定grid是否是高分辨率grid

3. **`_sample_grid_corners(grid, uv, resolution)`**: 手动实现4角点采样
   - 输入：UV坐标 [B, 2]
   - 输出：4个角点的特征 [B, 4 * feature_dim]

### 插值模式配置

- 高分辨率grid：使用 `"Nearest"` 插值（配合手动4角点采样）
- 低分辨率grid：使用 `"Linear"` 插值（tcnn自动双线性插值）

## 测试

运行测试脚本验证修改：

```bash
python test_corner_sampling.py
```

预期输出：
- 每个level的第一个grid（较高分辨率）输出16维特征（4角点×4维）
- 每个level的第二个grid（较低分辨率）输出4维特征（双线性）
- 总特征维度为80维

## 优势

1. **保留更多高频信息**：高分辨率grid不进行插值，保留原始网格值
2. **灵活性**：网络可以学习如何利用4个角点的信息
3. **渐进式细节**：低分辨率grid仍然使用插值提供平滑的基础信息
