"""
训练模块
包含HMM插件的训练基础设施

主要组件：
1. HMMTrainer: 核心训练器
2. TrainerConfig: 训练配置
3. 损失函数
4. 优化器和调度器
"""

from .trainer import HMMTrainer, TrainerConfig
from .loss import HMMLoss, create_hmm_loss
from .utils import save_checkpoint, load_checkpoint, setup_logging

__all__ = [
    'HMMTrainer',
    'TrainerConfig',
    'HMMLoss',
    'create_hmm_loss',
    'save_checkpoint',
    'load_checkpoint',
    'setup_logging'
] 