# FastSAM 葡萄实例分割

基于 [FastSAM](https://github.com/CASIA-LMC-Lab/FastSAM) 对葡萄图像进行实例分割，并将独立的葡萄颗粒合并为完整的葡萄串，同时保留茎部等细小结构的分割效果。

---

## 目录

- [问题描述](#问题描述)
- [解决方案](#解决方案)
- [环境配置](#环境配置)
- [文件说明](#文件说明)
- [运行方法](#运行方法)
- [关键参数说明](#关键参数说明)
- [技术细节](#技术细节)
- [结果说明](#结果说明)

---

## 问题描述

### 原始问题

FastSAM 在葡萄图像上进行分割时存在两个主要问题：

1. **颗粒过度细分**：每颗葡萄粒被独立分割为一个实例（约 150+ 个 mask），用户实际需要的是将整串葡萄作为一个整体区域。  
2. **茎部未被检测**：葡萄茎部细小，默认置信度阈值下容易被当作噪声过滤掉，导致茎部缺失。

### 目标

| 期望效果 | 解决方法 |
|---|---|
| 整串葡萄为一个 mask | 基于空间邻近度合并相邻颗粒 mask |
| 茎部被分割 | 调低置信度阈值，减少 NMS 过度抑制 |

---

## 解决方案

### 1. 参数调整（捕获茎部）

| 参数 | 默认值 | 调整后 | 说明 |
|---|---|---|---|
| `conf` | 0.4 | **0.25** | 降低置信度，茎部等弱置信目标不再被过滤 |
| `iou` | 0.9 | **0.7** | 降低 NMS IoU 阈值，相邻颗粒不被过度合并 |

### 2. 邻近合并算法

**核心思路**：将相邻的独立葡萄粒 mask 合并为一整串。

```
原始 N 个 mask
    ↓  对每个 mask 做膨胀（Dilation）
    ↓  所有膨胀后的 mask 叠加为一张二值图
    ↓  对叠加图做连通域分析（connectedComponents）
    ↓  原始 mask 按重心所在的连通域分组
    ↓  同组 mask 逻辑 OR 合并
  最终 M 个整体实例（M << N）
```

**效果**：
- 相互紧密的葡萄颗粒（膨胀后重叠）被归入同一连通域 → 合并为一串
- 不相邻的独立叶片、茎等保留为独立实例

**算法复杂度**：O(N × H × W)，线性于 mask 数量和图像尺寸，内存高效。

---

## 环境配置

### 前置要求

- NVIDIA GPU（≥ 4GB 显存，推荐 RTX 系列）
- NVIDIA 驱动 ≥ 525（支持 CUDA 12.x）
- Conda

### 创建虚拟环境

```bash
# 创建 Python 3.10 的 conda 环境
conda create -n fastsam_env python=3.10 -y
conda activate fastsam_env
```

### 安装 PyTorch（CUDA 12.1 版本，兼容 CUDA 12.x 驱动）

```bash
pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121
```

> **注意**：若驱动版本较旧（如支持 CUDA 11.x），请改用对应 cu118 版本。  
> 验证 CUDA 是否可用：
> ```bash
> python -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
> ```

### 安装其他依赖

```bash
pip install ultralytics opencv-python-headless matplotlib
```

---

## 文件说明

```
fastsam_test/
├── run_fastsam.py          # 主脚本（实例分割 + 邻近合并）
├── FastSAM-x.pt            # 模型权重（139 MB，自动下载）
├── test_1.jpg              # 输入图像（葡萄）
├── README.md               # 本文档
└── output/
    ├── result_raw.png      # ultralytics 原始逐粒分割图
    └── overlay_merged.png  # 三图对比（原图 / 合并分割 / 轮廓）
```

---

## 运行方法

```bash
cd fastsam_test
conda activate fastsam_env
python run_fastsam.py
```

### 输出说明

脚本运行后输出两张图像：

| 文件 | 内容 |
|---|---|
| `output/result_raw.png` | FastSAM 原始逐粒分割结果，每个葡萄粒独立上色 |
| `output/overlay_merged.png` | 三图对比：**原图** / **合并后分割**（整串着色）/ **轮廓描边图** |

---

## 关键参数说明

脚本顶部的配置块可按需调整：

```python
DEVICE        = "cuda"   # "cpu" 表示仅用 CPU（速度慢约 10x）
CONF          = 0.25     # 置信度阈值，越低越灵敏，可能增加噪声
IOU           = 0.7      # NMS IoU 阈值，越低越保留细节
DILATE_KERNEL = 25       # 膨胀核大小（px）：越大越容易合并相邻颗粒
MIN_AREA      = 500      # 最小 mask 面积（px²）：过滤噪声小块
```

### 调参建议

| 问题 | 建议 |
|---|---|
| 葡萄串依然碎成多块 | 增大 `DILATE_KERNEL`（30~50） |
| 不同葡萄串被过度合并 | 减小 `DILATE_KERNEL`（15~20） |
| 茎部依然未检测到 | 降低 `CONF`（0.15~0.2） |
| 输出噪声过多 | 增大 `MIN_AREA`（800~1500） |

---

## 技术细节

### GPU 兼容性注意事项

| 驱动 CUDA 版本 | 推荐 PyTorch 版本 |
|---|---|
| CUDA 12.x（驱动 ≥ 525） | `torch 2.4.x +cu121` |
| CUDA 11.8（驱动 ≥ 520） | `torch 2.2.x +cu118` |

**常见问题**：若 `torch.cuda.is_available()` 返回 `False`，通常是 PyTorch 编译版本与实际驱动不兼容，需要重装匹配版本的 torch。

### 内存优化

- 将 mask 从 `float32` 转为 `bool`，内存减半
- 推理完成后立即删除结果 tensor（`del result, results; gc.collect()`）
- 合并算法使用单一缓冲区复用（`np.empty` 预分配）
- GPU 推理时，masks 在 GPU 上生成后转 CPU，避免 CPU 内存溢出

### 合并算法内存分析

对于 N=189 个 mask，尺寸 768×1024：

| 数据 | bool 类型 | float32 类型 |
|---|---|---|
| 所有 masks | ~148 MB | ~596 MB |
| 合并缓冲区 | ~0.8 MB | ~3 MB |

使用 GPU 后，模型推理本身在显存中完成，CPU 内存主要用于 numpy 后处理，峰值约 300~500 MB，不会 OOM。

---

## 结果说明

### 运行输出（实测）

| 指标 | 数值 |
|---|---|
| 原始检测 mask 数 | 189 个 |
| 合并后实例数 | 3 个 |
| GPU 推理耗时 | 127 ms（RTX 2060）|
| 对比 CPU 推理 | ~2600 ms（约 20x 加速）|

### 合并后各实例面积

| 实例 | 面积（px） | 占比 | 描述 |
|---|---|---|---|
| 1 | 1,714,640 | 78.6% | 整体葡萄区域（所有葡萄串合并） |
| 2 | 1,612 | 0.1% | 小型孤立结构（茎/叶） |
| 3 | 3,613 | 0.2% | 小型孤立结构（茎/叶） |

> **注**：`DILATE_KERNEL=25` 将整张图中所有葡萄颗粒连通为一个大区域。若图中有多串独立葡萄且希望分别标注，可将该值减小至 10~15。

### 输出文件

运行成功后，`output/overlay_merged.png` 为三图对比：

- **左图（Original Image）**：输入的葡萄原图
- **中图（Merged Segmentation）**：合并后的实例，整串葡萄 / 叶片 / 茎各为独立区域，随机上色
- **右图（Contour Overlay）**：在原图上用彩色轮廓线标注各实例边界

`output/result_raw.png` 为 ultralytics 原始逐粒分割图（189 个独立颗粒，每粒独立上色）。
