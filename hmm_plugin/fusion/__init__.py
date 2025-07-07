"""
融合模块
包含权重学习和不同的融合策略

主要组件：
1. WeightLearner: 权重学习模块
2. FusionStrategy: 融合策略枚举
3. 各种融合策略的实现
"""

from .weight_learner import WeightLearner
from .fusion_strategies import FusionStrategy, create_fusion_strategy, create_fusion_module

__all__ = [
    'WeightLearner',
    'FusionStrategy', 
    'create_fusion_strategy',
    'create_fusion_module'
] 