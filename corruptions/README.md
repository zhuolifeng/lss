# 图像干扰和对抗攻击工具包

这个工具包提供了多种图像干扰方法和对抗攻击技术，用于测试和评估深度学习模型的鲁棒性。

## 文件结构

```
corruptions/
├── noise.py              # 基础干扰方法
├── attack.py             # 对抗攻击方法
├── corrupt.py            # 主整合类
├── attack_explanation.md # 攻击方法详细解释
└── README.md             # 本文件
```

## 安装依赖

```bash
# 基础依赖
pip install numpy opencv-python torch matplotlib
```

## 主要功能

### 1. 基础干扰方法 (`noise.py`)

提供三种基础图像干扰方法：

- **高斯噪声** (`gaussian_noise`): 添加正态分布随机噪声
- **曝光调整** (`exposure_adjustment`): 调整图像曝光度
- **图像模糊** (`blur_image`): 支持高斯、平均、中值模糊

### 2. 对抗攻击方法 (`attack.py`)

支持多种对抗攻击技术，所有方法都支持图像归一化：

#### 基于梯度的攻击
- **FGSM** (`fgsm_attack`): 快速梯度符号方法
- **PGD** (`pgd_attack`): 投影梯度下降
- **C&W** (`cw_attack`): Carlini & Wagner攻击
- **DeepFool** (`deepfool_attack`): 最小扰动攻击

#### 基于边界的攻击
- **Boundary Attack** (`boundary_attack`): 边界攻击

#### 基于变换的攻击
- **补丁攻击** (`patch_attack`): 添加对抗补丁
- **模糊攻击** (`blur_attack`): 高斯模糊
- **压缩攻击** (`compression_attack`): JPEG压缩
- **几何攻击** (`geometric_attack`): 旋转和缩放
- **不可见扰动攻击** (`invisible_perturbation_attack`): 人眼难以察觉的扰动

### 3. 主整合类 (`corrupt.py`)

`ImageCorruption` 类整合了所有干扰和攻击方法：

```python
from corruptions.corrupt import ImageCorruption

# 初始化
corruption = ImageCorruption(model=your_model, device='cpu')

# 应用单个方法
result = corruption.apply_corruption(image, 'fgsm', epsilon=0.1)

# 应用多个方法
methods = [
    ('gaussian', {'std': 20}),
    ('fgsm', {'epsilon': 0.1, 'normalize': True})
]
result = corruption.apply_multiple_corruptions(image, methods)

# 随机应用方法
result = corruption.apply_random_corruption(image, num_methods=3)
```

## 攻击方法详细说明

### 相似性
1. **目标一致**: 所有方法都旨在生成对抗样本
2. **输入输出**: 都接受原始图像，输出对抗样本
3. **归一化支持**: 都支持图像归一化，确保扰动不可见
4. **参数控制**: 都有控制攻击强度的参数

### 差异性

#### 计算复杂度
- **低复杂度**: FGSM, 补丁攻击, 模糊攻击, 压缩攻击, 几何攻击
- **中等复杂度**: PGD, DeepFool, 不可见扰动攻击
- **高复杂度**: C&W, Boundary Attack

#### 攻击能力
- **强攻击**: C&W, PGD, Boundary Attack
- **中等攻击**: FGSM, DeepFool
- **弱攻击**: 补丁攻击, 模糊攻击, 压缩攻击, 几何攻击

#### 扰动大小
- **小扰动**: DeepFool, C&W, 不可见扰动攻击
- **中等扰动**: FGSM, PGD
- **大扰动**: 补丁攻击, 模糊攻击, 几何攻击

#### 信息需求
- **需要梯度**: FGSM, PGD, C&W, DeepFool, 不可见扰动攻击
- **不需要梯度**: 补丁攻击, 模糊攻击, 压缩攻击, 几何攻击, Boundary Attack

## 使用示例

### 基础使用

```python
import cv2
import numpy as np
from corruptions.corrupt import ImageCorruption, apply_preset

# 加载图像
image = cv2.imread('test.jpg')

# 创建干扰实例
corruption = ImageCorruption()

# 应用高斯噪声
noisy_image = corruption.apply_corruption(image, 'gaussian', std=25)

# 应用预设
result = apply_preset(image, 'light_interference')
```

### 对抗攻击

```python
import torch
from corruptions.corrupt import ImageCorruption

# 定义模型（示例）
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 模型定义...
    
    def forward(self, x):
        # 前向传播...
        return output

# 初始化
model = SimpleModel()
corruption = ImageCorruption(model=model, device='cpu')

# FGSM攻击
adversarial_image = corruption.apply_corruption(
    image, 'fgsm', epsilon=0.1, normalize=True
)

# PGD攻击
adversarial_image = corruption.apply_corruption(
    image, 'pgd', epsilon=0.1, num_steps=20, normalize=True
)
```

## 预设配置

工具包提供了多种预设配置：

- `light_interference`: 轻微干扰
- `heavy_interference`: 重度干扰
- `adversarial_light`: 轻微对抗攻击
- `adversarial_heavy`: 重度对抗攻击
- `mixed_attack`: 混合攻击
- `invisible_attack`: 不可见攻击

## 选择建议

### 根据需求选择
1. **快速评估**: FGSM或PGD
2. **精确评估**: C&W或DeepFool
3. **黑盒攻击**: Boundary Attack或变换攻击
4. **物理攻击**: 补丁攻击或几何攻击
5. **隐蔽攻击**: 不可见扰动攻击

### 根据计算资源选择
1. **计算资源有限**: FGSM, 变换攻击
2. **计算资源充足**: PGD, C&W, Boundary Attack
3. **实时应用**: FGSM, 变换攻击

### 根据攻击目标选择
1. **模型鲁棒性评估**: C&W, DeepFool, PGD
2. **实际安全测试**: 变换攻击, 补丁攻击
3. **理论研究**: FGSM, Boundary Attack

## 注意事项

1. **模型要求**: 对抗攻击需要可微分的模型
2. **图像格式**: 输入图像应为numpy数组，格式为(H, W, C)
3. **归一化**: 建议使用归一化参数确保扰动不可见
4. **计算资源**: 某些攻击方法（如C&W）计算成本较高

## 扩展

可以通过继承 `NoiseCorruption` 或 `AdversarialAttack` 类来添加新的干扰或攻击方法。 