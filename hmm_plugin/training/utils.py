"""
HMM训练工具模块
提供训练过程中的辅助功能

主要组件：
1. AverageMeter: 平均值计算器
2. ProgressMeter: 进度显示器
3. 检查点保存和加载函数
4. 日志设置函数
"""

import os
import time
import json
import logging
from typing import Dict, List, Any, Optional, Union, Tuple

import torch
import numpy as np


class AverageMeter:
    """计算并存储平均值和当前值"""
    
    def __init__(self, name: str = '', fmt: str = ':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
    
    def reset(self):
        """重置所有计数器"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        """更新计数器"""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        """字符串表示"""
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    """进度显示器"""
    
    def __init__(self, num_batches: int, meters: list, prefix: str = ""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
    
    def display(self, batch: int):
        """显示进度"""
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
    
    def _get_batch_fmtstr(self, num_batches: int):
        """获取批次格式字符串"""
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def setup_logging(name: str, 
                  log_file: str, 
                  level: int = logging.INFO,
                  console: bool = True) -> logging.Logger:
    """
    设置日志
    
    Args:
        name: 日志器名称
        log_file: 日志文件路径
        level: 日志级别
        console: 是否输出到控制台
    
    Returns:
        logger: 日志器
    """
    # 创建日志器
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 创建日志文件目录
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # 文件处理器
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # 控制台处理器
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger


def set_seed(seed: int):
    """
    设置随机种子，保证结果可重复
    
    Args:
        seed: 随机种子
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(checkpoint: Dict[str, Any], 
                   checkpoint_dir: str, 
                   filename: str):
    """
    保存检查点
    
    Args:
        checkpoint: 检查点字典
        checkpoint_dir: 保存目录
        filename: 文件名
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")


def load_checkpoint(checkpoint_path: str, 
                   device: torch.device) -> Dict[str, Any]:
    """
    加载检查点
    
    Args:
        checkpoint_path: 检查点路径
        device: 设备
    
    Returns:
        checkpoint: 检查点字典
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print(f"Loaded checkpoint from {checkpoint_path}")
    return checkpoint


def count_parameters(model: torch.nn.Module) -> int:
    """
    计算模型参数数量
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_memory_usage() -> Dict[str, float]:
    """
    获取GPU显存使用情况
    
    Returns:
        memory_dict: 显存使用字典
    """
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
        memory_reserved = torch.cuda.memory_reserved() / (1024 ** 3)    # GB
        max_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)    # GB
        
        return {
            'allocated': memory_allocated,
            'reserved': memory_reserved,
            'max': max_memory
        }
    else:
        return {'allocated': 0, 'reserved': 0, 'max': 0}


def save_config(config: Dict[str, Any], config_path: str):
    """保存配置到JSON文件"""
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to {config_path}")


def load_config(config_path: str) -> Dict[str, Any]:
    """从JSON文件加载配置"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


class EarlyStopping:
    """早停机制"""
    
    def __init__(self, 
                 patience: int = 7, 
                 min_delta: float = 0, 
                 restore_best_weights: bool = True):
        """
        初始化
        
        Args:
            patience: 容忍轮数
            min_delta: 最小变化量
            restore_best_weights: 是否恢复最佳权重
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_weights = None
    
    def __call__(self, val_loss: float, model: torch.nn.Module) -> bool:
        """
        检查是否早停
        
        Args:
            val_loss: 验证损失
            model: 模型
        
        Returns:
            early_stop: 是否早停
        """
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            if self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
        else:
            self.best_score = score
            if self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
        
        return self.early_stop


class LearningRateScheduler:
    """学习率调度器包装器"""
    
    def __init__(self, 
                 optimizer: torch.optim.Optimizer,
                 mode: str = 'min',
                 factor: float = 0.1,
                 patience: int = 10,
                 min_lr: float = 0,
                 eps: float = 1e-08):
        """
        初始化
        
        Args:
            optimizer: 优化器
            mode: 模式，'min'或'max'
            factor: 缩放因子
            patience: 容忍轮数
            min_lr: 最小学习率
            eps: 最小变化量
        """
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.eps = eps
        
        self.best_score = None
        self.counter = 0
        
        # 验证模式
        assert mode in ['min', 'max'], "mode must be 'min' or 'max'"
        
        # 记录初始学习率
        self.init_lrs = [group['lr'] for group in optimizer.param_groups]
    
    def step(self, metrics: float):
        """
        调度器步进
        
        Args:
            metrics: 指标值
        """
        # 初始化最佳分数
        if self.best_score is None:
            self.best_score = metrics
            return
        
        # 检查是否需要减少学习率
        if (self.mode == 'min' and metrics < self.best_score - self.eps) or \
           (self.mode == 'max' and metrics > self.best_score + self.eps):
            self.best_score = metrics
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self._reduce_lr()
                self.counter = 0
    
    def _reduce_lr(self):
        """减少学习率"""
        for i, group in enumerate(self.optimizer.param_groups):
            old_lr = group['lr']
            new_lr = max(old_lr * self.factor, self.min_lr)
            if old_lr - new_lr > self.eps:
                group['lr'] = new_lr
                print(f'Reducing learning rate of group {i} from {old_lr:.6f} to {new_lr:.6f}')


class MetricTracker:
    """指标跟踪器"""
    
    def __init__(self):
        """初始化"""
        self.metrics = {}
    
    def update(self, **kwargs):
        """
        更新指标
        
        Args:
            **kwargs: 指标字典
        """
        for key, value in kwargs.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)
    
    def avg(self, key: str, last_n: int = None) -> float:
        """
        计算指标平均值
        
        Args:
            key: 指标名称
            last_n: 最后n个值，None表示所有
            
        Returns:
            avg: 平均值
        """
        values = self.metrics.get(key, [])
        if not values:
            return 0.0
        
        if last_n is not None:
            values = values[-last_n:]
        
        return sum(values) / len(values)
    
    def get(self, key: str) -> list:
        """获取指标值列表"""
        return self.metrics.get(key, [])
    
    def reset(self):
        """重置所有指标"""
        self.metrics = {}
    
    def summary(self) -> Dict[str, float]:
        """
        获取所有指标的平均值
        
        Returns:
            summary_dict: 摘要字典
        """
        return {
            key: sum(values) / len(values) if values else 0.0
            for key, values in self.metrics.items()
        }


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: tuple = (1,)) -> list:
    """
    计算topk准确率
    
    Args:
        output: 预测输出 [batch_size, num_classes]
        target: 目标标签 [batch_size]
        topk: topk值元组
    
    Returns:
        res: topk准确率列表
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].float().sum()
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def compute_iou(pred: torch.Tensor, target: torch.Tensor, 
                num_classes: int, ignore_index: int = -1) -> torch.Tensor:
    """
    计算IoU
    
    Args:
        pred: 预测结果 [N, H, W]
        target: 目标标签 [N, H, W]
        num_classes: 类别数
        ignore_index: 忽略的索引
    
    Returns:
        iou: 每个类别的IoU [num_classes]
    """
    with torch.no_grad():
        # 创建mask
        mask = (target != ignore_index)
        
        # 将预测和目标展平
        pred_flat = pred[mask]
        target_flat = target[mask]
        
        # 计算混淆矩阵
        confusion_matrix = torch.zeros(num_classes, num_classes, device=pred.device)
        for i in range(num_classes):
            for j in range(num_classes):
                confusion_matrix[i, j] = torch.sum((pred_flat == i) & (target_flat == j))
        
        # 计算IoU
        iou = torch.zeros(num_classes, device=pred.device)
        for i in range(num_classes):
            intersection = confusion_matrix[i, i]
            union = torch.sum(confusion_matrix[i, :]) + torch.sum(confusion_matrix[:, i]) - intersection
            if union > 0:
                iou[i] = intersection / union
        
        return iou


def create_optimizer(model: torch.nn.Module, 
                    optimizer_type: str = 'adamw',
                    learning_rate: float = 1e-4,
                    weight_decay: float = 1e-5,
                    **kwargs) -> torch.optim.Optimizer:
    """
    创建优化器
    
    Args:
        model: 模型
        optimizer_type: 优化器类型
        learning_rate: 学习率
        weight_decay: 权重衰减
        **kwargs: 其他参数
    
    Returns:
        optimizer: 优化器
    """
    if optimizer_type == 'adam':
        return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, **kwargs)
    elif optimizer_type == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, **kwargs)
    elif optimizer_type == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def warmup_lr_scheduler(optimizer: torch.optim.Optimizer,
                       warmup_iters: int,
                       warmup_factor: float = 1.0 / 3):
    """预热学习率调度器"""
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f) 