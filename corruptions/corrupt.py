import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple, Union, Callable
import random
from .noise import NoiseCorruption
from .attack import AdversarialAttack
from .attack_classic import ClassicAdversarialAttack


class ImageCorruption:
    """图像干扰主类 - 整合干扰和攻击方法"""
    
    def __init__(self, model: Optional[Callable] = None ):
        """
        初始化图像干扰类
        
        Args:
            model: 目标模型 (用于对抗攻击)
        """
        self.noise_corruption = NoiseCorruption()
        self.adversarial_attack = AdversarialAttack(model)
        self.classic_adversarial_attack = ClassicAdversarialAttack(model)
    
    def apply_noise_corruption(self, images, intensity):
        """
        随机选择3张图像应用噪声干扰
        
        Args:
            images: 输入图像 (S, C, H, W) torch.Tensor, S=6
            intensity: 干扰强度 'low', 'medium', 'high' 或具体参数字典
            
        Returns:
            应用干扰后的图像 (S, C, H, W) torch.Tensor
        """
        S, C, H, W = images.shape
        
        # 随机选择3张图像的索引
        selected_indices = random.sample(range(S), 3)
        
        for i in selected_indices:
            images[i] = self.noise_corruption.apply_random_interference(images[i], intensity=intensity)
        
        return images

    def apply_attack_corruption(self, images, rots, trans, intrins, post_rots, post_trans, binimgs, intensity):
        """
        应用对抗攻击到图像上 - 随机选择3张图像进行攻击
        
        Args:
            images: 输入图像 (S, C, H, W) torch.Tensor, S=6
            rots, trans, intrins, post_rots, post_trans, binimgs: 对应的参数
            intensity: 攻击强度 'low', 'medium', 'high' 或具体参数字典
            
        Returns:
            应用攻击后的图像 (S, C, H, W) torch.Tensor (3张被攻击,3张保持原样)
        """
        return self.adversarial_attack.apply_random_attack(images, rots, trans, intrins, post_rots, post_trans, binimgs, intensity=intensity)
        #return self.classic_adversarial_attack.apply_random_attack(images, rots, trans, intrins, post_rots, post_trans, binimgs, intensity=intensity)
    
    def apply_corruption(self, images, rots, trans, intrins, post_rots, post_trans, binimgs, type, intensity):
        """
        应用图像干扰
        
        Args:
            images: 输入图像 (B, S, C, H, W)
            type: 干扰类型 'noise' 或 'attack'
            intensity: 干扰强度
                - 'low': 轻微干扰
                - 'medium': 中等干扰  
                - 'high': 强烈干扰
                - dict: 自定义参数
        """
        B, S, C, H, W = images.shape
        print(images.shape)
        for i in range(B):
            if type == 'noise':
                images[i] = self.apply_noise_corruption(images[i], intensity=intensity)
            elif type == 'attack':
                #pass
                images[i] = self.apply_attack_corruption(images[i], rots[i], trans[i], intrins[i], post_rots[i], post_trans[i], binimgs[i], intensity=intensity)
            else:
                raise ValueError(f"Invalid corruption type: {type}")
        return images
