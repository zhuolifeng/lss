import numpy as np
import cv2
from typing import Tuple, Optional, Union, Callable
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import random


class AdversarialAttack:
    """对抗攻击类 - 支持图像归一化,处理H,W,C格式图像"""
    
    def __init__(self, model: Optional[Callable] = None, device: str = 'cpu'):
        """
        初始化攻击类
        
        Args:
            model: 目标模型
            device: 计算设备
        """
        self.model = model
        self.device = device
        
    def _check_image_format(self, image: torch.Tensor):
        """检查图像格式是否为H,W,C"""
        if len(image.shape) != 3:
            raise ValueError(f"输入图像必须是3维 (H, W, C)，当前形状: {image.shape}")
        
    def normalize_image(self, image: torch.Tensor, target_range: Tuple[float, float] = (0, 1)) -> torch.Tensor:
        """
        归一化图像到指定范围
        
        Args:
            image: 输入图像 (H, W, C) torch.Tensor
            target_range: 目标范围 (min, max)
            
        Returns:
            归一化后的图像 torch.Tensor
        """
        self._check_image_format(image)
        min_val, max_val = target_range
        normalized = (image.float() - image.min()) / (image.max() - image.min())
        normalized = normalized * (max_val - min_val) + min_val
        return normalized
    
    def denormalize_image(self, image: torch.Tensor, original_range: Tuple[float, float] = (0, 1)) -> torch.Tensor:
        """
        反归一化图像
        
        Args:
            image: 归一化后的图像 torch.Tensor
            original_range: 原始范围
            
        Returns:
            反归一化后的图像 (H, W, C) torch.Tensor
        """
        min_val, max_val = original_range
        denormalized = (image - min_val) / (max_val - min_val)
        denormalized = denormalized * 255
        return torch.clamp(denormalized, 0, 255)
    
    def fgsm_attack(self, images: torch.Tensor, rots: torch.Tensor, trans: torch.Tensor, 
                    intrins: torch.Tensor, post_rots: torch.Tensor, post_trans: torch.Tensor, binimgs: torch.Tensor,
                    epsilon: float = 0.1, normalize: bool = True) -> torch.Tensor:
        """
        Fast Gradient Sign Method (FGSM) 攻击 - 随机选择3张图像进行攻击
        
        Args:
            images: 输入图像序列 (S, H, W, C) torch.Tensor, S=6
            rots, trans, intrins, post_rots, post_trans, binimgs: 相机参数 (S, ...)
            epsilon: 扰动大小
            normalize: 是否归一化扰动
            
        Returns:
            对抗样本序列 (S, H, W, C) torch.Tensor
        """
        S, H, W, C = images.shape
        
        if self.model is None:
            raise ValueError("模型未设置,无法进行FGSM攻击")
        
        # 随机选择3张图像的索引
        selected_indices = random.sample(range(S), 3)
        
        # 归一化图像到[0,1]范围
        if normalize:
            images_normalized = torch.stack([self.normalize_image(images[i], (0, 1)) for i in range(S)])
        else:
            images_normalized = images.float() / 255.0
        
        # 转换为模型输入格式 (B=1, S=6, C, H, W)
        images_tensor = images_normalized.permute(0, 3, 1, 2).unsqueeze(0)
        images_tensor.requires_grad_(True)
        
        # 添加批次维度到所有参数 (B=1, S=6, ...)
        rots_batch = rots.unsqueeze(0)
        trans_batch = trans.unsqueeze(0) 
        intrins_batch = intrins.unsqueeze(0)
        post_rots_batch = post_rots.unsqueeze(0)
        post_trans_batch = post_trans.unsqueeze(0)
        # 前向传播
        output = self.model(images_tensor, rots_batch, trans_batch, 
                           intrins_batch, post_rots_batch, post_trans_batch)
        

        # 使用binimgs作为真实标签
        binimgs_batch = binimgs.unsqueeze(0).to(self.device)
        loss = F.binary_cross_entropy_with_logits(output, binimgs_batch.float())
        
        # 反向传播
        loss.backward()
        
        # 计算扰动，但只对选中的图像应用
        perturbation = epsilon * images_tensor.grad.sign()
        
        # 创建掩码，只对选中的图像应用扰动
        mask = torch.zeros_like(perturbation)
        for idx in selected_indices:
            mask[0, idx, :, :, :] = 1.0
        
        # 应用掩码
        perturbation = perturbation * mask
        
        # 生成对抗样本
        adversarial_images = images_tensor + perturbation
        
        # 裁剪到有效范围
        adversarial_images = torch.clamp(adversarial_images, 0, 1)
        
        # 转换回S,H,W,C格式，移除批次维度
        adversarial_shwc = adversarial_images.squeeze(0).permute(0, 2, 3, 1)
        if normalize:
            return torch.stack([self.denormalize_image(adversarial_shwc[i], (0, 1)) for i in range(S)])
        else:
            return adversarial_shwc * 255
    
    def pgd_attack(self, images: torch.Tensor, rots: torch.Tensor, trans: torch.Tensor,
                   intrins: torch.Tensor, post_rots: torch.Tensor, post_trans: torch.Tensor, binimgs: torch.Tensor,
                   epsilon: float = 0.1, alpha: float = 0.01, num_steps: int = 20, 
                normalize: bool = True) -> torch.Tensor:
        """
        Projected Gradient Descent (PGD) 攻击 - 随机选择3张图像进行攻击
        
        Args:
            images: 输入图像序列 (S, H, W, C) torch.Tensor, S=6
            rots, trans, intrins, post_rots, post_trans, binimgs: 相机参数 (S, ...)
            epsilon: 扰动大小
            alpha: 步长
            num_steps: 迭代步数
            normalize: 是否归一化扰动
            
        Returns:
            对抗样本序列 (S, H, W, C) torch.Tensor
        """
        S, H, W, C = images.shape
        
        if self.model is None:
            raise ValueError("模型未设置,无法进行PGD攻击")
        
        # 随机选择3张图像的索引
        selected_indices = random.sample(range(S), 3)
        
        # 归一化图像到[0,1]范围
        if normalize:
            images_normalized = torch.stack([self.normalize_image(images[i], (0, 1)) for i in range(S)])
        else:
            images_normalized = images.float() / 255.0
        
        # 转换为模型输入格式 (B=1, S=6, C, H, W)
        images_tensor = images_normalized.permute(0, 3, 1, 2).unsqueeze(0)
        
        # 添加批次维度到所有参数 (B=1, S=6, ...)
        rots_batch = rots.unsqueeze(0)
        trans_batch = trans.unsqueeze(0) 
        intrins_batch = intrins.unsqueeze(0)
        post_rots_batch = post_rots.unsqueeze(0)
        post_trans_batch = post_trans.unsqueeze(0)
        
        # 创建掩码，只对选中的图像应用扰动
        mask = torch.zeros((1, S, 1, 1, 1)).to(images_tensor.device)
        for idx in selected_indices:
            mask[0, idx, 0, 0, 0] = 1.0
        mask = mask.expand_as(images_tensor)
        
        # 初始化随机扰动，但只对选中的图像应用
        delta = (torch.rand_like(images_tensor) * 2 * epsilon - epsilon) * mask
        
        for _ in range(num_steps):
            delta.requires_grad_(True)
            
            # 前向传播
            output = self.model(images_tensor + delta, rots_batch, trans_batch,
                               intrins_batch, post_rots_batch, post_trans_batch)
            
            # 使用binimgs作为真实标签
            binimgs_batch = binimgs.unsqueeze(0).to(self.device)
            loss = F.binary_cross_entropy_with_logits(output, binimgs_batch.float())
            
            # 反向传播
            loss.backward()
            
            # 更新扰动，但只对选中的图像应用梯度
            grad_update = alpha * delta.grad.sign() * mask
            delta = delta + grad_update
            
            # 投影到epsilon球内，但只对选中的图像
            delta = torch.clamp(delta, -epsilon, epsilon) * mask
            
            # 裁剪到有效范围
            delta = (torch.clamp(images_tensor + delta, 0, 1) - images_tensor) * mask
            
        adversarial_images = images_tensor + delta
        
        # 转换回S,H,W,C格式，移除批次维度
        adversarial_shwc = adversarial_images.squeeze(0).permute(0, 2, 3, 1)
        if normalize:
            return torch.stack([self.denormalize_image(adversarial_shwc[i], (0, 1)) for i in range(S)])
        else:
            return adversarial_shwc * 255
    

    def apply_random_attack(self, images, rots, trans, intrins, post_rots, post_trans, binimgs) -> torch.Tensor:
        """
        随机选择3张图像应用攻击方法,掩蔽其余图像梯度
        
        Args:
            images: 输入图像 (S, H, W, C) torch.Tensor, S=6
            rots, trans, intrins, post_rots, post_trans, binimgs: 模型所需的其他参数
            
        Returns:
            应用攻击后的图像 (S, H, W, C) torch.Tensor (3张被攻击,3张保持原样)
        """
        # 随机选择攻击类型（目前支持FGSM和PGD）
        attack_type = random.choice(['fgsm', 'pgd'])
        
        if attack_type == 'fgsm':
            return self.fgsm_attack(images, rots, trans, intrins, post_rots, post_trans, binimgs, epsilon=random.uniform(0.05, 0.2))
        elif attack_type == 'pgd':
            return self.pgd_attack(images, rots, trans, intrins, post_rots, post_trans, binimgs, epsilon=random.uniform(0.05, 0.2), num_steps=random.randint(10, 30))
        else:
            return images
