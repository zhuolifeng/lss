"""
训练工具函数
包含检查点保存/加载、日志设置、平均值计算等功能
"""

import os
import logging
import torch
import numpy as np
import random
from typing import Dict, Any, Optional
import json
from pathlib import Path


class AverageMeter:
    """计算和存储平均值和当前值"""
    
    def __init__(self, name: str = '', fmt: str = ':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    """进度显示器"""
    
    def __init__(self, num_batches: int, meters: list, prefix: str = ""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
    
    def display(self, batch: int):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(m) for m in self.meters]
        print('\t'.join(entries))
    
    def _get_batch_fmtstr(self, num_batches: int):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def setup_logging(name: str, 
                  log_file: str, 
                  level: int = logging.INFO,
                  console: bool = True) -> logging.Logger:
    """设置日志记录器"""
    
    # 创建日志目录
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # 创建logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 避免重复添加handler
    if logger.handlers:
        return logger
    
    # 创建formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 文件handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # 控制台handler
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger


def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(checkpoint: Dict[str, Any], 
                   checkpoint_dir: str, 
                   filename: str):
    """保存检查点"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(checkpoint_path: str, 
                   device: torch.device) -> Dict[str, Any]:
    """加载检查点"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print(f"Checkpoint loaded from {checkpoint_path}")
    return checkpoint


def count_parameters(model: torch.nn.Module) -> int:
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_memory_usage() -> Dict[str, float]:
    """获取GPU内存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        cached = torch.cuda.memory_reserved() / 1024**3      # GB
        return {
            'allocated': allocated,
            'cached': cached,
            'free': cached - allocated
        }
    else:
        return {'allocated': 0, 'cached': 0, 'free': 0}


def save_config(config: Dict[str, Any], config_path: str):
    """保存配置文件"""
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    with open(config_path, 'r') as f:
        return json.load(f)


class EarlyStopping:
    """早停机制"""
    
    def __init__(self, 
                 patience: int = 7, 
                 min_delta: float = 0, 
                 restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss: float, model: torch.nn.Module) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_weights = model.state_dict().copy() if self.restore_best_weights else None
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_weights = model.state_dict().copy() if self.restore_best_weights else None
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        return False


class LearningRateScheduler:
    """学习率调度器"""
    
    def __init__(self, 
                 optimizer: torch.optim.Optimizer,
                 mode: str = 'min',
                 factor: float = 0.1,
                 patience: int = 10,
                 min_lr: float = 0,
                 eps: float = 1e-08):
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.eps = eps
        
        self.best = None
        self.num_bad_epochs = 0
        self.last_epoch = 0
        
    def step(self, metrics: float):
        current = float(metrics)
        
        if self.best is None:
            self.best = current
        elif self.mode == 'min' and current < self.best - self.eps:
            self.best = current
            self.num_bad_epochs = 0
        elif self.mode == 'max' and current > self.best + self.eps:
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
        
        if self.num_bad_epochs > self.patience:
            self._reduce_lr()
            self.num_bad_epochs = 0
        
        self.last_epoch += 1
    
    def _reduce_lr(self):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lr)
            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr
                print(f'Reducing learning rate of group {i} to {new_lr:.4e}.')


class MetricTracker:
    """指标跟踪器"""
    
    def __init__(self):
        self.data = {}
        
    def update(self, **kwargs):
        for key, value in kwargs.items():
            if key not in self.data:
                self.data[key] = []
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.data[key].append(value)
    
    def avg(self, key: str, last_n: int = None) -> float:
        """计算指定指标的平均值"""
        if key not in self.data:
            return 0.0
        values = self.data[key]
        if last_n:
            values = values[-last_n:]
        return sum(values) / len(values) if values else 0.0
    
    def get(self, key: str) -> list:
        """获取指定指标的所有值"""
        return self.data.get(key, [])
    
    def reset(self):
        """重置所有数据"""
        self.data = {}
    
    def summary(self) -> Dict[str, float]:
        """获取所有指标的摘要"""
        summary = {}
        for key, values in self.data.items():
            if values:
                summary[f'{key}_avg'] = sum(values) / len(values)
                summary[f'{key}_last'] = values[-1]
                summary[f'{key}_min'] = min(values)
                summary[f'{key}_max'] = max(values)
        return summary


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: tuple = (1,)) -> list:
    """计算top-k准确率"""
    maxk = max(topk)
    batch_size = target.size(0)
    
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def compute_iou(pred: torch.Tensor, target: torch.Tensor, 
                num_classes: int, ignore_index: int = -1) -> torch.Tensor:
    """计算IoU"""
    pred = pred.flatten()
    target = target.flatten()
    
    # 忽略指定类别
    mask = target != ignore_index
    pred = pred[mask]
    target = target[mask]
    
    iou_per_class = []
    for c in range(num_classes):
        pred_c = (pred == c)
        target_c = (target == c)
        
        intersection = (pred_c & target_c).sum().float()
        union = (pred_c | target_c).sum().float()
        
        if union == 0:
            iou_per_class.append(float('nan'))
        else:
            iou_per_class.append((intersection / union).item())
    
    return torch.tensor(iou_per_class)


def create_optimizer(model: torch.nn.Module, 
                    optimizer_type: str = 'adamw',
                    learning_rate: float = 1e-4,
                    weight_decay: float = 1e-5,
                    **kwargs) -> torch.optim.Optimizer:
    """创建优化器"""
    params = [p for p in model.parameters() if p.requires_grad]
    
    if optimizer_type.lower() == 'adamw':
        return torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay, **kwargs)
    elif optimizer_type.lower() == 'adam':
        return torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay, **kwargs)
    elif optimizer_type.lower() == 'sgd':
        return torch.optim.SGD(params, lr=learning_rate, weight_decay=weight_decay, 
                              momentum=kwargs.get('momentum', 0.9), **kwargs)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def warmup_lr_scheduler(optimizer: torch.optim.Optimizer,
                       warmup_iters: int,
                       warmup_factor: float = 1.0 / 3):
    """学习率预热调度器"""
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def get_device_info() -> Dict[str, Any]:
    """获取设备信息"""
    info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda,
        'device_count': torch.cuda.device_count(),
    }
    
    if torch.cuda.is_available():
        info['device_name'] = torch.cuda.get_device_name()
        info['memory_allocated'] = torch.cuda.memory_allocated() / 1024**3
        info['memory_reserved'] = torch.cuda.memory_reserved() / 1024**3
    
    return info


def cleanup_old_files(directory: str, pattern: str, keep_last: int = 5):
    """清理旧文件"""
    if not os.path.exists(directory):
        return
    
    files = [f for f in os.listdir(directory) if pattern in f]
    files.sort(key=lambda x: os.path.getmtime(os.path.join(directory, x)))
    
    if len(files) > keep_last:
        for f in files[:-keep_last]:
            os.remove(os.path.join(directory, f))
            print(f"Removed old file: {f}")


# 示例使用
if __name__ == "__main__":
    # 测试平均值计算器
    meter = AverageMeter('Loss', ':.4f')
    for i in range(10):
        meter.update(np.random.random())
    print(f"Average meter: {meter}")
    
    # 测试指标跟踪器
    tracker = MetricTracker()
    for i in range(5):
        tracker.update(
            loss=np.random.random(),
            accuracy=np.random.random() * 100,
            lr=1e-4 * (0.9 ** i)
        )
    
    print("Metric tracker summary:")
    for key, value in tracker.summary().items():
        print(f"  {key}: {value:.4f}")
    
    # 测试设备信息
    print("\nDevice info:")
    device_info = get_device_info()
    for key, value in device_info.items():
        print(f"  {key}: {value}")
    
    # 测试内存使用
    print("\nMemory usage:")
    memory_info = get_memory_usage()
    for key, value in memory_info.items():
        print(f"  {key}: {value:.2f} GB") 