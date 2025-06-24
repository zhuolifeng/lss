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
            image: 输入图像 (C, H, W) torch.Tensor
            mean: 噪声均值
            std: 噪声标准差
            
        Returns:
            添加噪声后的图像 (C, H, W) torch.Tensor
        """
        # 确保输入是H,W,C格式
        if len(image.shape) != 3:
            raise ValueError(f"输入图像必须是3维 (C, H, W)，当前形状: {image.shape}")
        
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
            image: 输入图像 (C, H, W) torch.Tensor
            factor: 曝光调整因子 (0.1-3.0)
            
        Returns:
            调整曝光后的图像 (C, H, W) torch.Tensor
        """
        # 确保输入是H,W,C格式
        if len(image.shape) != 3:
            raise ValueError(f"输入图像必须是3维 (C, H, W)，当前形状: {image.shape}")
        
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
            image: 输入图像 (C, H, W) torch.Tensor
            kernel_size: 模糊核大小 (必须是奇数)
            
        Returns:
            模糊后的图像 (C, H, W) torch.Tensor
        """
        # 确保输入是H,W,C格式
        if len(image.shape) != 3:
            raise ValueError(f"输入图像必须是3维 (C, H, W)，当前形状: {image.shape}")
        
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
    
    def apply_random_interference(self, image: torch.Tensor, intensity) -> torch.Tensor:
        """
        随机应用一种干扰方法
        
        Args:
            image: 输入图像 (C, H, W) torch.Tensor
            intensity: 干扰强度
                - 'minimal': 极轻微干扰 (noise_std: 0.5-1.5, exposure: 0.98-1.02)
                - 'very_low': 很轻微干扰 (noise_std: 1.0-2.5, exposure: 0.95-1.05)
                - 'low': 轻微干扰 (noise_std: 1.5-3.5, exposure: 0.92-1.08)
                - 'medium_low': 中低等干扰 (noise_std: 3.0-6.0, exposure: 0.88-1.12)
                - 'medium': 中等干扰 (noise_std: 5.0-10.0, exposure: 0.8-1.2)
                - 'medium_high': 中高等干扰 (noise_std: 8.0-15.0, exposure: 0.75-1.3)
                - 'high': 强烈干扰 (noise_std: 12.0-25.0, exposure: 0.6-1.6)
                - 'extreme': 极强干扰 (noise_std: 20.0-50.0, exposure: 0.3-2.5)
                - dict: 自定义参数 {'noise_std': float, 'exposure_factor': float, 'blur_kernel': int}
            
        Returns:
            应用干扰后的图像 (C, H, W) torch.Tensor
        """
        # 确保输入是H,W,C格式
        if len(image.shape) != 3:
            raise ValueError(f"输入图像必须是3维 (C, H, W)，当前形状: {image.shape}")
        
        # 定义强度配置 - 调整为更精细的强度控制，减少对IoU的影响
        intensity_configs = {
            'minimal': {
                'noise_std_range': (0.5, 1.5),
                'exposure_range': (0.98, 1.02),
                'blur_kernel_range': (3, 3)
            },
            'very_low': {
                'noise_std_range': (1.0, 2.5),
                'exposure_range': (0.95, 1.05),
                'blur_kernel_range': (3, 3)
            },
            'low': {
                'noise_std_range': (1.5, 3.5),
                'exposure_range': (0.92, 1.08),
                'blur_kernel_range': (3, 3)
            },
            'medium_low': {
                'noise_std_range': (3.0, 6.0),
                'exposure_range': (0.88, 1.12),
                'blur_kernel_range': (3, 5)
            },
            'medium': {
                'noise_std_range': (5.0, 10.0),
                'exposure_range': (0.8, 1.2),
                'blur_kernel_range': (3, 5)
            },
            'medium_high': {
                'noise_std_range': (8.0, 15.0),
                'exposure_range': (0.75, 1.3),
                'blur_kernel_range': (3, 7)
            },
            'high': {
                'noise_std_range': (12.0, 25.0),
                'exposure_range': (0.6, 1.6),
                'blur_kernel_range': (5, 9)
            },
            'extreme': {
                'noise_std_range': (20.0, 50.0),
                'exposure_range': (0.3, 2.5),
                'blur_kernel_range': (7, 11)
            }
        }
        
        # 获取配置
        if isinstance(intensity, str):
            if intensity not in intensity_configs:
                raise ValueError(f"不支持的强度级别: {intensity}，支持: {list(intensity_configs.keys())}")
            config = intensity_configs[intensity]
        elif isinstance(intensity, dict):
            config = intensity
        else:
            raise ValueError("intensity必须是字符串('low'/'medium'/'high')或字典")
        
        interference_type = random.choice(['gaussian', 'exposure', 'blur'])
        
        if interference_type == 'gaussian':
            if 'noise_std' in config:
                std = config['noise_std']
            else:
                std = random.uniform(*config['noise_std_range'])
            return self.gaussian_noise(image, std=std)
            
        elif interference_type == 'exposure':
            if 'exposure_factor' in config:
                factor = config['exposure_factor']
            else:
                factor = random.uniform(*config['exposure_range'])
            return self.exposure_adjustment(image, factor=factor)
            
        elif interference_type == 'blur':
            if 'blur_kernel' in config:
                kernel_size = config['blur_kernel']
            else:
                kernel_range = config['blur_kernel_range']
                # 确保选择奇数
                kernel_options = [k for k in range(kernel_range[0], kernel_range[1] + 1) if k % 2 == 1]
                kernel_size = random.choice(kernel_options)
            return self.blur_image(image, kernel_size=kernel_size)
            
        else:
            raise ValueError(f"不支持的干扰类型: {interference_type}")
    
    def test_intensity_levels(self, image: torch.Tensor) -> dict:
        """
        测试所有强度级别的效果
        
        Args:
            image: 输入图像 (C, H, W) torch.Tensor
            
        Returns:
            dict: 包含所有强度级别结果的字典
        """
        results = {}
        levels = ['minimal', 'very_low', 'low', 'medium_low', 'medium', 'medium_high', 'high', 'extreme']
        
        for level in levels:
            try:
                # 使用固定的噪声类型来保证一致性
                corrupted = self.gaussian_noise(
                    image, 
                    std=self._get_noise_std_for_level(level)
                )
                results[level] = {
                    'image': corrupted,
                    'noise_std': self._get_noise_std_for_level(level),
                    'max_diff': torch.abs(corrupted - image).max().item(),
                    'mean_diff': torch.abs(corrupted - image).mean().item()
                }
            except Exception as e:
                results[level] = {'error': str(e)}
        
        return results
    
    def _get_noise_std_for_level(self, level: str) -> float:
        """获取特定强度级别的噪声标准差中值"""
        intensity_configs = {
            'minimal': (0.5, 1.5),
            'very_low': (1.0, 2.5),
            'low': (1.5, 3.5),
            'medium_low': (3.0, 6.0),
            'medium': (5.0, 10.0),
            'medium_high': (8.0, 15.0),
            'high': (12.0, 25.0),
            'extreme': (20.0, 50.0)
        }
        
        if level not in intensity_configs:
            raise ValueError(f"未知强度级别: {level}")
        
        min_std, max_std = intensity_configs[level]
        return (min_std + max_std) / 2.0
    
    def apply_custom_noise(self, image: torch.Tensor, noise_std: float = 3.0, 
                          exposure_factor: float = 1.0, blur_kernel: int = 3) -> torch.Tensor:
        """
        应用自定义参数的噪声
        
        Args:
            image: 输入图像 (C, H, W) torch.Tensor
            noise_std: 高斯噪声标准差 (推荐范围: 1-10 for minimal impact)
            exposure_factor: 曝光调整因子 (推荐范围: 0.9-1.1 for minimal impact)
            blur_kernel: 模糊核大小 (推荐: 3 for minimal impact)
            
        Returns:
            处理后的图像 (C, H, W) torch.Tensor
        """
        # 应用高斯噪声
        noisy_image = self.gaussian_noise(image, std=noise_std)
        
        # 应用曝光调整
        if exposure_factor != 1.0:
            noisy_image = self.exposure_adjustment(noisy_image, factor=exposure_factor)
        
        # 应用模糊（如果需要）
        if blur_kernel > 3:
            noisy_image = self.blur_image(noisy_image, kernel_size=blur_kernel)
        
        return noisy_image
