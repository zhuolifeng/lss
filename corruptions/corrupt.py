import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple, Union, Callable
import random
from .noise import NoiseCorruption
from .attack import AdversarialAttack
from .attack_classic import ClassicAdversarialAttack
from .dag_attack import DAGBEVAttack
from .fusion_attack import FusionAttack
from .robust_bev_attack import RobustBEVAttack


class ImageCorruption:
    """图像干扰主类 - 整合干扰和攻击方法"""
    
    def __init__(self, model: Optional[Callable] = None ):
        """
        初始化图像干扰类
        
        Args:
            model: 目标模型 (用于对抗攻击)
        """
        self.noise_corruption = NoiseCorruption()
        self.classic_adversarial_attack = ClassicAdversarialAttack(model)
        self.dag_bev_attack = DAGBEVAttack(model)
        self.fusion_attack = FusionAttack(model)
        self.robust_bev_attack = RobustBEVAttack(model)
    
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

    def apply_fgsm_attack(self, images, rots, trans, intrins, post_rots, post_trans, binimgs, intensity):
        """应用FGSM攻击"""
        return self.classic_adversarial_attack.apply_fgsm_attack(images, rots, trans, intrins, post_rots, post_trans, binimgs, intensity=intensity)
    
    def apply_pgd_attack(self, images, rots, trans, intrins, post_rots, post_trans, binimgs, intensity):
        """应用PGD攻击"""
        return self.classic_adversarial_attack.apply_pgd_attack(images, rots, trans, intrins, post_rots, post_trans, binimgs, intensity=intensity)
    
    def apply_cw_attack(self, images, rots, trans, intrins, post_rots, post_trans, binimgs, intensity):
        """应用C&W攻击"""
        return self.classic_adversarial_attack.apply_cw_attack(images, rots, trans, intrins, post_rots, post_trans, binimgs, intensity=intensity)
    
    def apply_dag_attack(self, images, rots, trans, intrins, post_rots, post_trans, binimgs, intensity):
        """应用DAG攻击"""
        return self.dag_bev_attack.apply_dag_attack(images, rots, trans, intrins, post_rots, post_trans, binimgs, intensity=intensity)
    
    def apply_fusion_attack(self, images, rots, trans, intrins, post_rots, post_trans, binimgs, intensity):
        """应用Fusion攻击"""
        return self.fusion_attack.apply_fusion_attack(images, rots, trans, intrins, post_rots, post_trans, binimgs, intensity=intensity)
    
    def apply_robust_bev_attack(self, images, rots, trans, intrins, post_rots, post_trans, binimgs, intensity):
        """应用RobustBEV攻击"""
        return self.robust_bev_attack.apply_robust_bev_attack(images, rots, trans, intrins, post_rots, post_trans, binimgs, intensity=intensity)
    
    def apply_corruption(self, images, rots, trans, intrins, post_rots, post_trans, binimgs, type, intensity):
        """
        应用图像干扰
        
        Args:
            images: 输入图像 (B, S, C, H, W)
            type: 干扰类型 
                - 'noise': 噪声干扰
                - 'fgsm', 'pgd', 'cw': 经典对抗攻击
                - 'dag': DAG攻击
                - 'fusion': Fusion攻击 (需要指定attack_type)
                - 'robust_bev': RobustBEV攻击 (需要指定attack_type)
                - 'origin': 无攻击
            intensity: 干扰强度
                - 'low': 轻微干扰
                - 'medium': 中等干扰  
                - 'high': 强烈干扰
                - dict: 自定义参数
            **kwargs: 额外参数，如attack_type等
        """
        B, S, C, H, W = images.shape
        print(f"处理图像形状: {images.shape}")
        
        for i in range(B):
            if type == 'noise':
                images[i] = self.apply_noise_corruption(images[i], intensity=intensity)
            elif type == 'fgsm':
                images[i] = self.apply_fgsm_attack(images[i], rots[i], trans[i], intrins[i], post_rots[i], post_trans[i], binimgs[i], intensity=intensity)
            elif type == 'pgd':
                images[i] = self.apply_pgd_attack(images[i], rots[i], trans[i], intrins[i], post_rots[i], post_trans[i], binimgs[i], intensity=intensity)
            elif type == 'cw':
                images[i] = self.apply_cw_attack(images[i], rots[i], trans[i], intrins[i], post_rots[i], post_trans[i], binimgs[i], intensity=intensity)
            elif type == 'dag':
                images[i] = self.apply_dag_attack(images[i], rots[i], trans[i], intrins[i], post_rots[i], post_trans[i], binimgs[i], intensity=intensity)
            elif type == 'fusion':
                images[i] = self.apply_fusion_attack(images[i], rots[i], trans[i], intrins[i], post_rots[i], post_trans[i], binimgs[i], intensity=intensity)
            elif type == 'robust_bev':
                images[i] = self.apply_robust_bev_attack(images[i], rots[i], trans[i], intrins[i], post_rots[i], post_trans[i], binimgs[i], intensity=intensity)
            elif type == 'origin':
                pass
            else:
                raise ValueError(f"Invalid corruption type: {type}")
        return images
