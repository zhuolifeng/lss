"""
融合策略模块
定义融合模块的创建工厂函数
"""

from typing import Union
import torch.nn as nn

from .weight_learner import WeightLearner, TripleFusion


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
    if strategy_type == 'weight_learner':
        return create_weight_learner(lift_dim, splat_dim, hidden_dim)
    elif strategy_type == 'triple_fusion':
        return create_triple_fusion(splat_dim, hidden_dim)
    else:
        raise ValueError(f"Unknown fusion strategy: {strategy_type}") 