"""
HMM插件 - 可插拔的隐式马尔可夫链模块
用于增强各种LSS模型的时序建模能力

主要组件：
1. HMM核心插件
2. Diffusion-based Transition1 (UNet, DiT, Mamba)
3. ControlNet-based Transition2
4. 融合策略模块
5. 权重学习网络
"""

from .hmm_core import HMMPlugin, HMMConfig
from .transitions import (
    DiffusionTransition1, 
    ControlNetTransition2,
    BaseTransition
)
from .fusion import WeightLearner, FusionStrategy
from .models import UNetDiffusion, DiTDiffusion, MambaDiffusion
from .utils import ActionEncoder

__version__ = "1.0.0"
__author__ = "HMM-BEV Team"

__all__ = [
    "HMMPlugin",
    "HMMConfig", 
    "DiffusionTransition1",
    "ControlNetTransition2",
    "BaseTransition",
    "WeightLearner",
    "FusionStrategy",
    "UNetDiffusion",
    "DiTDiffusion", 
    "MambaDiffusion",
    "ActionEncoder"
] 