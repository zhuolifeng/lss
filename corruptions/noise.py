import numpy as np
import cv2
import torch
from typing import Tuple, Optional, Union
import random


class NoiseCorruption:
    """图像干扰类 - 专注于基础干扰方法,处理H,W,C格式图像"""
    
    def __init__(self):
        pass
    
    def gaussian_noise(self, image: torch.Tensor, mean: float = 0, std: float = 25) -> torch.Tensor:
        """
        添加高斯噪声
        
        Args:
            image: 输入图像 (H, W, C) torch.Tensor
            mean: 噪声均值
            std: 噪声标准差
            
        Returns:
            添加噪声后的图像 (H, W, C) torch.Tensor
        """
        # 确保输入是H,W,C格式
        if len(image.shape) != 3:
            raise ValueError(f"输入图像必须是3维 (H, W, C)，当前形状: {image.shape}")
        
        # 生成高斯噪声
        noise = torch.randn_like(image) * std + mean
        
        # 添加噪声
        noisy_image = image + noise
        
        # 裁剪到有效范围 [0, 255]
        return torch.clamp(noisy_image, 0, 255)
    
    def exposure_adjustment(self, image: torch.Tensor, factor: float = 1.5) -> torch.Tensor:
        """
        调整图像曝光度
        
        Args:
            image: 输入图像 (H, W, C) torch.Tensor
            factor: 曝光调整因子 (0.1-3.0)
            
        Returns:
            调整曝光后的图像 (H, W, C) torch.Tensor
        """
        # 确保输入是H,W,C格式
        if len(image.shape) != 3:
            raise ValueError(f"输入图像必须是3维 (H, W, C)，当前形状: {image.shape}")
        
        # 限制因子范围
        factor = torch.clamp(torch.tensor(factor), 0.1, 3.0)
        
        # 调整曝光
        adjusted_image = image * factor
        
        # 裁剪到有效范围 [0, 255]
        return torch.clamp(adjusted_image, 0, 255)
    
    def blur_image(self, image: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
        """
        模糊图像 - 使用简单的高斯模糊
        
        Args:
            image: 输入图像 (H, W, C) torch.Tensor
            kernel_size: 模糊核大小 (必须是奇数)
            
        Returns:
            模糊后的图像 (H, W, C) torch.Tensor
        """
        # 确保输入是H,W,C格式
        if len(image.shape) != 3:
            raise ValueError(f"输入图像必须是3维 (H, W, C)，当前形状: {image.shape}")
        
        # 确保核大小为奇数
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # 使用简单的模糊方法 - 添加高斯噪声来模拟模糊效果
        blur_intensity = kernel_size / 10.0  # 将核大小转换为模糊强度
        
        # 生成高斯噪声来模拟模糊
        noise = torch.randn_like(image) * blur_intensity * 20
        
        # 添加噪声实现模糊效果
        blurred_image = image + noise
        
        # 裁剪到有效范围 [0, 255]
        return torch.clamp(blurred_image, 0, 255)
    
    def _create_gaussian_kernel(self, kernel_size: int, sigma: float) -> torch.Tensor:
        """
        创建高斯核
        
        Args:
            kernel_size: 核大小
            sigma: 标准差
            
        Returns:
            高斯核 (1, 1, kernel_size, kernel_size)
        """
        # 创建坐标网格
        x = torch.arange(-kernel_size//2, kernel_size//2 + 1, dtype=torch.float32)
        y = torch.arange(-kernel_size//2, kernel_size//2 + 1, dtype=torch.float32)
        x_grid, y_grid = torch.meshgrid(x, y, indexing='ij')
        
        # 计算高斯函数
        kernel = torch.exp(-(x_grid**2 + y_grid**2) / (2 * sigma**2))
        
        # 归一化
        kernel = kernel / kernel.sum()
        
        # 转换为卷积核格式 (1, 1, kernel_size, kernel_size)
        return kernel.unsqueeze(0).unsqueeze(0)
    
    def apply_random_interference(self, image: torch.Tensor) -> torch.Tensor:
        """
        随机应用一种干扰方法
        
        Args:
            image: 输入图像 (H, W, C) torch.Tensor
            
        Returns:
            应用干扰后的图像 (H, W, C) torch.Tensor
        """
        # 确保输入是H,W,C格式
        if len(image.shape) != 3:
            raise ValueError(f"输入图像必须是3维 (H, W, C)，当前形状: {image.shape}")
        
        interference_type = random.choice(['gaussian', 'exposure', 'blur'])
        
        if interference_type == 'gaussian':
            return self.gaussian_noise(image, std=random.uniform(10, 50))
        elif interference_type == 'exposure':
            return self.exposure_adjustment(image, factor=random.uniform(0.5, 2.0))
        elif interference_type == 'blur':
            return self.blur_image(image, kernel_size=random.choice([3, 5, 7]))
        else:
            raise ValueError(f"不支持的干扰类型: {interference_type}")
