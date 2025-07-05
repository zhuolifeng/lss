"""
训练模块
包含HMM插件的训练基础设施

主要组件：
1. HMMTrainer: 核心训练器
2. TrainerConfig: 训练配置
3. 数据加载器
4. 损失函数
5. 优化器和调度器
"""

from .trainer import HMMTrainer, TrainerConfig
from .data_loader import HMMDataLoader, create_data_loaders
from .loss_functions import HMMLoss, create_loss_function
from .utils import save_checkpoint, load_checkpoint, setup_logging

__all__ = [
    'HMMTrainer',
    'TrainerConfig',
    'HMMDataLoader',
    'create_data_loaders',
    'HMMLoss',
    'create_loss_function',
    'save_checkpoint',
    'load_checkpoint',
    'setup_logging'
] 