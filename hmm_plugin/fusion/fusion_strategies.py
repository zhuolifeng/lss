"""
融合策略模块
定义融合模块的创建工厂函数
"""

from enum import Enum
from typing import Union
import torch.nn as nn

from .weight_learner import WeightLearner, TripleFusion


class FusionStrategy(Enum):
    """融合策略枚举"""
    WEIGHT_LEARNER = 'weight_learner'
    TRIPLE_FUSION = 'triple_fusion'
    MULTI_PATH = 'multi_path'
    HIERARCHICAL = 'hierarchical'
    ADAPTIVE = 'adaptive'


def create_weight_learner(
    lift_dim: int,
    splat_dim: int,
    hidden_dim: int = 256
) -> nn.Module:
    """
    创建权重学习模块
    
    用于学习如何融合 Xt + Xt' → final Xt 和 Yt + Yt' → fused Yt
    
    Args:
        lift_dim: lift特征维度
        splat_dim: splat特征维度
        hidden_dim: 隐藏层维度
        
    Returns:
        weight_learner: 权重学习模块
    """
    return WeightLearner(
        lift_dim=lift_dim,
        splat_dim=splat_dim,
        hidden_dim=hidden_dim
    )


def create_triple_fusion(
    splat_dim: int,
    hidden_dim: int = 256
) -> nn.Module:
    """
    创建三路融合模块
    
    用于融合 Yt + Yt' + Yt'' → final Yt
    
    Args:
        splat_dim: splat特征维度
        hidden_dim: 隐藏层维度
        
    Returns:
        triple_fusion: 三路融合模块
    """
    return TripleFusion(
        splat_dim=splat_dim,
        hidden_dim=hidden_dim
    )


def create_fusion_strategy(
    strategy_type: str,
    lift_dim: int = 64,
    splat_dim: int = 64,
    hidden_dim: int = 256
) -> nn.Module:
    """
    创建融合策略的统一接口
    
    Args:
        strategy_type: 融合策略类型 ('weight_learner' 或 'triple_fusion')
        lift_dim: lift特征维度
        splat_dim: splat特征维度
        hidden_dim: 隐藏层维度
        
    Returns:
        fusion_module: 对应的融合模块
    """
    if strategy_type == FusionStrategy.WEIGHT_LEARNER.value:
        return create_weight_learner(lift_dim, splat_dim, hidden_dim)
    elif strategy_type == FusionStrategy.TRIPLE_FUSION.value:
        return create_triple_fusion(splat_dim, hidden_dim)
    else:
        raise ValueError(f"Unknown fusion strategy: {strategy_type}")


def create_fusion_module(
    config,
    lift_dim: int = 64,
    splat_dim: int = 64
) -> nn.Module:
    """
    根据配置创建融合模块
    
    Args:
        config: 融合配置
        lift_dim: lift特征维度
        splat_dim: splat特征维度
        
    Returns:
        fusion_module: 融合模块
    """
    strategy = config.get('strategy', 'weight_learner')
    hidden_dim = config.get('hidden_dim', 256)
    
    if strategy == 'multi_path':
        return create_triple_fusion(splat_dim, hidden_dim)
    else:
        return create_weight_learner(lift_dim, splat_dim, hidden_dim) 