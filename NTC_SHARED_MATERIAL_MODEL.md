# NTC Shared Material Model

## 目标结构

每个材质使用一套共享的 feature-grid 金字塔和一套 MLP，同时解码
basecolor、metalness、normal 和 roughness，而不是为每种纹理分别训练
一套 grid 和 MLP。

```text
UV + LOD
   |
   +-- 8x8 tile partition and local-UV positional encoding = 12D
   +-- large grid: 4 surrounding feature vectors x 12D = 48D
   +-- small grid: bilinear-interpolated feature vector = 12D
   +-- LOD parameter = 1D
   |
   +-- MLP input = 12 + 48 + 12 + 1 = 73D
           |
           +-- hidden layer 0: 64D + hardGELU
           +-- hidden layer 1: 64D + hardGELU
           +-- output: 8D, linear (no activation)
```

最终 MLP 结构为：

```text
73 -> 64 -> 64 -> 8
```

## 位置编码：8x8 Tile 局部坐标

位置编码不直接使用整张纹理的全局 UV，而是先把纹理划分为 `8x8` 个 tile，再在每个 tile 内计算局部坐标：

```glsl
vec2 tileUv = fract(uv) * 8.0;
vec2 tileIndex = floor(tileUv);
vec2 localUv = fract(tileUv);
```

其中：

- `tileIndex` 表示当前像素属于哪个 tile；
- `localUv` 是当前像素在 tile 内的局部坐标，范围为 `[0, 1)`；
- 位置编码只对 `localUv` 计算；
- 所有 tile 使用相同的局部位置编码，因此编码模式会在不同 tile 中周期性重复。

tile index 不作为当前 MLP 的额外输入；材质的空间差异由 feature grid 特征表达，局部位置编码负责表达 tile 内的位置变化。

对 `localUv` 使用 5 组三角波频率，并为水平和垂直方向各追加一个固定常数项：

```glsl
float triangleWave(float x) {
    return 2.0 * abs(x - floor(x + 0.5));
}

for (int i = 0; i < 5; ++i) {
    vec2 scaled = localUv * exp2(float(i));
    x[i * 2 + 0] = triangleWave(scaled.x);
    x[i * 2 + 1] = triangleWave(scaled.y);
}

x[10] = 1.0; // horizontal constant
x[11] = 1.0; // vertical constant
```

因此位置编码仍然是：

```text
Horizontal: 5 个变化特征 + 1 个常数 = 6D
Vertical:   5 个变化特征 + 1 个常数 = 6D
总计：12D
```

频率为：

```text
1、2、4、8、16
```

最后两个常数项不编码位置，作用是提供 DC/基础分量，类似显式 bias 输入；常数值固定为 `1.0`，对应的 MLP 权重仍然参与训练。
这里的 `8x8` 指纹理 tile 的空间划分，不是位置编码的维度。

## Grid level 采样方式

每个 grid level 包含两个逻辑 grid：

```text
Level L
  +-- large grid: 12 feature channels
  +-- small grid: 12 feature channels
```

### Large grid

large grid 不使用普通双线性插值，而是读取当前 UV 周围的四个角点：

```text
large00: 12D
large10: 12D
large01: 12D
large11: 12D
```

拼接得到 32D：

```text
4 x 12 = 48D
```

### Small grid

small grid 对当前 UV 进行双线性插值，得到一个 12D feature vector：

```text
small = bilinearSample(smallGrid, uv)
```

因此一个 grid level 贡献：

```text
48D large-grid features + 12D small-grid features = 60D
```

## MLP 输入排列

输入向量必须在 Python 训练、导出器、CPU reference 和 GLSL 中保持完全一致：

```text
x[0..4]    : horizontal triangle-wave features (5D)
x[5]       : horizontal constant (1D, fixed 1.0)
x[6..10]   : vertical triangle-wave features (5D)
x[11]      : vertical constant (1D, fixed 1.0)
x[12..23]  : large00 (12D)
x[24..35]  : large10 (12D)
x[36..47]  : large01 (12D)
x[48..59]  : large11 (12D)
x[60..71]  : small bilinear feature (12D)
x[72]      : normalized LOD
```

因此：

```text
12D positional encoding
+ 60D grid features
+ 1D LOD
= 73D MLP input
```

建议在 Python 和 shader 中都增加输入维度断言，防止通道排列或 grid 数量变化后静默错位。

## MLP 输出通道

## MLP 激活函数

两个隐藏层的线性变换之后都使用论文中的分段 `hardGELU`：

```text
hardGELU(x) = 0,                 if x < -3/2
              x,                 if x >  3/2
              (x / 3) * (x+3/2), otherwise
```

对应实现：

```glsl
float hardGelu(float x) {
    if (x < -1.5) return 0.0;
    if (x >  1.5) return x;
    return (x / 3.0) * (x + 1.5);
}
```

激活函数只应用于两个隐藏层：

```text
h0 = hardGELU(W0 * input + b0)
h1 = hardGELU(W1 * h0    + b1)
output = W2 * h1 + b2
```

输出层保持线性，不使用 `hardGELU`、sigmoid、tanh 或 clamp。输出通道的范围约束和材质语义转换在 MLP 输出之后单独处理。

输出固定为 8 个逻辑通道：

```text
output[0]   : basecolor.r
output[1]   : basecolor.g
output[2]   : basecolor.b
output[3]   : metalness
output[4]   : normal.r
output[5]   : normal.g
output[6]   : normal.b
output[7]   : roughness
```

其中：

- basecolor 在训练前应转换到线性空间；
- normal 以 `[0, 1]` 编码训练，运行时转换到 `[-1, 1]`；
- metalness 和 roughness 是单通道标量；
- normal 进入渲染 G-buffer 前仍需结合 TBN 矩阵重建切线空间法线。

第一版不包含 AO、emissive、alpha 或 displacement；这些通道以后可以扩展输出维度，但必须同步更新训练目标、metadata、CPU reference 和 GLSL。

## Feature grid 的物理存储

逻辑上每个 grid 有 8 个 feature channels。由于当前 RGBA 纹理每张只能保存 4 个通道，物理存储可以拆成两张纹理：

```text
part0: feature 0..3
part1: feature 4..7
```

Python、CPU reference 和 GLSL 采样后必须重新拼成一个 12D vector。拆分只是存储布局，不改变模型的逻辑 feature dimension。

每个逻辑 level 仍然有 large 和 small 两类 grid，因此物理纹理数量由：

```text
grid levels x 2 grid types x 2 RGBA parts
```

决定。

## 量化和权重布局

feature grid 继续支持量化感知训练和 fake quantization/STE。导出和 Vulkan 解码必须使用相同的：

- 量化 bit 数；
- 数值范围；
- round/clamp 规则；
- 反量化公式。

MLP 权重不能依赖旧版固定常量 `37`、`48`、`32`、`3072`。导出的 metadata 应记录：

```text
logical_input_dims = 73
hidden_width = 64
num_hidden_layers = 2
output_width = 8
physical matrix stride
bias location and layout
```

如果 tiny-cuda-nn 对矩阵进行了对齐，必须区分逻辑维度和物理存储维度；不能把物理 padding 误认为新的逻辑输入。

## 论文式 16 bit 标量量化训练流程

本项目采用论文中的 simulated quantization，并使用 16 bit scalar quantization。

### 量化参数

```text
B = 16
N = 2^16 = 65536
Q = 1 / N = 1 / 65536
```

每个 grid 的量化范围为：

```text
minRange = -((N - 1) / 2) * Q ≈ -0.49999237
maxRange =  (N / 2) * Q =  0.5
```

量化步长约为：

```text
Q ≈ 0.0000152588
```

非对称范围让零值处于量化 bin 的中心，避免零值因为落在 bin 边界而产生额外误差。

### 训练前向：模拟量化噪声

feature grid 在训练内部仍然使用 FP32。每次 forward 对 feature 添加均匀噪声：

```text
noise ~ Uniform(-Q / 2, Q / 2)
featureForMlp = featureGrid + noise
```

16 bit 时噪声范围为：

```text
[-0.0000076294, 0.0000076294]
```

噪声加在 feature grid 上，而不是加在最终的 basecolor、normal、metalness 或 roughness 输出上。

### 反向更新后的范围约束

训练步骤的顺序固定为：

```text
加均匀噪声
  -> MLP forward
  -> 计算重建 loss
  -> backward
  -> optimizer.step()
  -> 将原始 feature grid clamp 到 [minRange, maxRange]
```

不能只在 forward 输入上 clamp。必须在 optimizer 更新完成后限制真正的 feature 参数，防止它们漂出最终可量化范围。

### 训练末期显式离散化

接近训练结束时，停止添加噪声，并把 feature grid 显式量化到 65536 个离散等级：

```text
index = round((feature - minRange) / Q)
index = clamp(index, 0, 65535)
featureQuantized = minRange + index * Q
```

显式量化后，feature grid 被冻结，只继续训练 MLP 约原计划步数的 5%，让 MLP 适应最终实际部署的离散 feature。

```text
feature grid: frozen
MLP weights: continue optimizing for approximately 5% extra steps
```

### 导出格式

导出的是已经显式量化后的 `uint16` feature index。若使用 RGBA8 图像保存 16 bit 值，则拆成高低字节：

```text
highByte = (index >> 8) & 0xff
lowByte  = index & 0xff
```

一个逻辑 8 通道 grid 可以拆为两个 RGBA part：

```text
part0: feature 0..3
part1: feature 4..7
```

每个 part 再保存对应的 high/low byte 图像。拆分只改变物理存储，不改变逻辑上的 12D feature vector。

也可以使用 `VK_FORMAT_R16G16B16A16_UNORM` 直接保存 16 bit 量化值；两种方式的逻辑量化规则必须完全一致。

MLP 权重暂时保持 FP32，不与 feature grid 使用同一套量化策略。

### Vulkan 推理反量化

Vulkan shader 从 high/low byte 图像读取并重组：

```text
value = (highByte << 8) | lowByte
feature = minRange + value * Q
```

反量化公式、字节顺序、`Q`、`minRange` 和 Python 导出代码必须完全一致。

之后按固定顺序构造 73D 输入：

```text
large00, large10, large01, large11, small
  -> 60D grid features
  -> 12D tile-local triangle-wave encoding
  -> 1D LOD
  -> 73D MLP input
```

### 量化一致性检查

训练、导出和 Vulkan 推理必须共享以下参数和规则：

```text
quantization bits = 16
N = 65536
Q = 1 / 65536
minRange / maxRange
round and clamp rules
high/low byte order
feature channel order
large-grid corner order
small-grid bilinear interpolation
MLP weight layout
```

推荐导出 CPU reference 数据：

```text
input_73.bin
hidden0_64.bin
hidden1_64.bin
output_8.bin
```

用于逐层比较 Python、CPU reference 和 GLSL 的数值，确认量化、feature 采样和 MLP 权重布局没有偏差。

## Python 训练接口要求

## 训练采样策略

训练不把整张 4K 纹理一次性送入网络，而是使用局部随机 crop：

```text
每个 batch：8 个随机 256x256 texel crop
```

一个 batch 内的 crop 使用同一个 detail level/LOD。每个 crop 从对应 LOD 的真实材质纹理中取监督值，同时使用当前选中的 grid level 构造 NTC 输入：

```text
随机选择 8 个 256x256 crop
  -> 选择一个 LOD/grid level
  -> large grid 四角采样 4 个 12D feature vector
  -> small grid 双线性采样 1 个 12D feature vector
  -> 拼接 60D grid feature
  -> 加入 12D tile-local positional encoding
  -> 加入 1D LOD
  -> 得到 73D MLP 输入
```

### LOD 指数分布

主要的 95% batch 使用指数分布选择 LOD：

```text
X ~ Uniform(0, 1)
LOD = floor(-log4(X))
```

等价的 Python 写法为：

```python
x = torch.rand(())
lod = torch.floor(-torch.log(x) / math.log(4.0)).long()
lod = lod.clamp(0, max_lod)
```

该分布偏向低数值 LOD，也就是高分辨率、细节更多的 mip level。理论上未截断时：

```text
P(LOD = 0) = 3/4
P(LOD = 1) = 3/16
P(LOD = 2) = 3/64
...
```

### 低分辨率 LOD 的均匀补采样

指数分布可能导致低分辨率 mip level 采样不足，因此另外 5% 的 batch 在整个 mip 链范围内均匀选择 LOD：

```python
if torch.rand(()) < 0.05:
    lod = torch.randint(0, max_lod + 1, ())
else:
    x = torch.rand(())
    lod = torch.floor(-torch.log(x) / math.log(4.0)).long()
    lod = lod.clamp(0, max_lod)
```

这样训练预算主要用于高分辨率细节，同时保证低分辨率 grid level 也能得到监督。

### 学习率和优化器

feature grid 和 MLP 使用同一个 Adam 优化器的两个 parameter group，但采用不同初始学习率：

```python
optimizer = torch.optim.Adam([
    {"params": feature_grid_parameters, "lr": 0.01},
    {"params": mlp_parameters, "lr": 0.005},
])
```

两组学习率都使用 cosine annealing，在训练结束时逐步降到 0。量化训练阶段仍按前文执行噪声模拟和更新后 clamp；feature grid 显式量化并冻结后，只继续使用 Adam 微调 MLP 约 5% 的额外步数。

## 重建损失函数

训练使用论文推荐的 L2 重建损失，也就是对 NTC 输出材质向量和真实材质向量之间的平方误差进行平均：

```text
L2 = mean((prediction - target)^2)
```

对于每个 batch，损失在以下维度上求平均：

```text
batch samples
256x256 crop 内的 texels
8 个材质输出通道
```

对应的 PyTorch 形式为：

```python
prediction = model(input_vector)  # [..., 8]
loss = torch.mean((prediction - target) ** 2)
```

其中 `target` 的通道顺序固定为：

```text
basecolor.r, basecolor.g, basecolor.b,
metalness,
normal.r, normal.g, normal.b,
roughness
```

监督值和网络输出必须使用相同的数值空间：

- basecolor 使用线性空间值计算 L2；
- normal 使用 `[0, 1]` 编码值计算 L2，推理时再转换到 `[-1, 1]`；
- metalness 和 roughness 使用归一化标量计算 L2。

第一版使用统一的等权 L2 损失，以便直接复现论文式基础训练流程。若后续发现 basecolor 的动态范围压制了 normal 或材质标量的学习，可以在保持 L2 形式不变的前提下加入固定的 per-channel 权重：

```python
channel_weights = torch.tensor([
    1.0, 1.0, 1.0,  # basecolor
    0.5,            # metalness
    1.0, 1.0, 1.0,  # normal
    0.5,            # roughness
], device=prediction.device)

loss = torch.mean(
    (prediction - target) ** 2 * channel_weights
)
```

每个材质的监督目标为一个 8D 向量：

```python
target = torch.cat([
    basecolor_rgb,  # 3D
    metalness,      # 1D
    normal_rgb,     # 3D
    roughness,      # 1D
], dim=-1)
```

训练优化器固定使用 Adam，学习率使用余弦退火调度：

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=initial_learning_rate,
    betas=(0.9, 0.999),
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=total_training_steps,
    eta_min=minimum_learning_rate,
)
```

每次完成一次 `optimizer.step()` 后调用 `scheduler.step()`。量化训练阶段仍按前文流程在更新后 clamp feature grid；末期显式量化并冻结 feature grid 后，只使用 Adam 继续微调 MLP，余弦调度继续沿用剩余的 5% 微调步数。

训练模型应使用：

```text
input dimensions: 73
hidden width: 64
hidden layers: 2
output dimensions: 8
hidden activation: hardGELU
output activation: none (linear)
```

建议对不同材质通道设置独立 loss 权重，避免 basecolor 的数值范围完全主导 normal、metalness 和 roughness 的训练。

## 导出格式要求

每个材质只导出一套共享资源：

```text
material_xxx/
  metadata.json
  level_*_large_part*_hi.png
  level_*_large_part*_lo.png
  level_*_small_part*_hi.png
  level_*_small_part*_lo.png
  mlp_params.bin
```

metadata 至少记录：

```json
{
  "n_input_dims": 73,
  "hidden_dim": 64,
  "num_hidden_layers": 2,
  "output_dim": 8,
  "grid_feature_dim": 12,
  "large_samples_per_level": 4,
  "small_samples_per_level": 1,
  "positional_encoding_dims": 12,
  "lod_dims": 1,
  "channel_order": [
    "basecolor_r", "basecolor_g", "basecolor_b",
    "metalness",
    "normal_r", "normal_g", "normal_b",
    "roughness"
  ]
}
```

## Vulkan 推理接口

每个 material slot 只绑定该材质的一套共享 grid 和一套 MLP 权重。compute shader 对每个像素只执行一次 decoder：

```text
feedback UV/material slot
  -> large grid 四角采样
  -> small grid 双线性采样
  -> 拼接 73D input
  -> 73 -> 64 -> 64 -> 8 MLP
  -> 写入 albedo、normal、material G-buffer
```

解码结果映射为：

```text
output[0..2] -> gbufferAlbedo
output[3]    -> metalness
output[4..6] -> tangent-space normal reconstruction
output[7]    -> roughness
```

Python、CPU reference 和 GLSL 必须使用相同的 feature 采样顺序、量化公式、矩阵布局和输出通道顺序。

## CPU/GPU 验证要求

重构后应导出并比较：

```text
input_73.bin
hidden0_64.bin
hidden1_64.bin
output_8.bin
```

验证顺序：

1. Python forward 与 CPU reference 数值一致；
2. CPU grid sampling 与 GLSL grid sampling 一致；
3. CPU MLP 每层输出与 GLSL 每层输出一致；
4. basecolor、normal、metalness、roughness 分别与原始纹理对比；
5. 检查不同 LOD level 之间的过渡；
6. 比较量化前后的重建误差；
7. 记录 Vulkan decode 的 GPU 时间和显存占用。

## 与旧模型的差异

旧模型：

```text
37 → 32 → 32 → 11
```

旧模型包含 4 通道 feature grid，并输出 canonical 11 通道材质数据。

新模型：

```text
73 → 64 → 64 → 8
```

新模型中每个 level 的 large/small grid 都是 8 通道，large grid 使用四角离散采样，small grid 使用双线性采样，并且一次只输出 Sponza 4K 当前需要的四类材质数据。

## 后续扩展

后续可以在本文档继续补充：

- Python 训练代码的具体文件和接口改动；
- grid 配置和导出文件命名规范；
- Vulkan descriptor 与 shader 常量定义；
- CPU/GPU 逐层数值验证脚本；
- Sponza 4K 的训练参数、显存预算和性能目标。
