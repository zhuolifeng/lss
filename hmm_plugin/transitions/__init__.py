"""
Transition模块
包含各种类型的transition函数，用于预测特征的变化量

主要组件：
1. BaseTransition: 基础transition类
2. DiffusionTransition1: 基于diffusion的transition1，预测delta_X
3. ControlNetTransition2: 基于ControlNet的transition2，预测delta_Y
"""

from .base_transition import BaseTransition
from .diffusion_transition import DiffusionTransition1
from .controlnet_transition import ControlNetTransition2

__all__ = [
    'BaseTransition',
    'DiffusionTransition1', 
    'ControlNetTransition2'
] 