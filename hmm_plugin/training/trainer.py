"""
HMM训练器模块
实现HMM插件的训练流程

核心组件：
1. TrainerConfig: 训练配置类
2. HMMTrainer: HMM训练器
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import os
import time
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union, Tuple
from tqdm import tqdm

from ..hmm_core import HMMPlugin


@dataclass
class TrainerConfig:
    """训练配置类"""
    # 基础配置
    batch_size: int = 16
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    
    # 设备
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 调度器
    scheduler_type: str = "cosine"  # "step", "cosine", "linear", "plateau"
    scheduler_params: Dict[str, Any] = None
    
    # 梯度裁剪
    grad_clip: bool = True
    grad_clip_norm: float = 1.0
    
    # 日志和保存
    log_dir: str = "./logs"
    checkpoint_dir: str = "./checkpoints"
    log_every: int = 10
    save_every: int = 5
    eval_every: int = 1
    
    # 早停
    early_stopping: bool = True
    patience: int = 10
    min_delta: float = 0.001
    
    # 混合精度训练
    use_amp: bool = False
    
    def __post_init__(self):
        """初始化后处理"""
        if self.scheduler_params is None:
            if self.scheduler_type == "cosine":
                self.scheduler_params = {"T_max": self.epochs, "eta_min": 1e-6}
            elif self.scheduler_type == "step":
                self.scheduler_params = {"step_size": 30, "gamma": 0.1}
            elif self.scheduler_type == "plateau":
                self.scheduler_params = {"mode": "min", "factor": 0.1, "patience": 5}
            else:
                self.scheduler_params = {}


class HMMTrainer:
    """HMM训练器 - 训练HMM插件"""
    
    def __init__(self, 
                 model: HMMPlugin,
                 train_loader: DataLoader,
                 val_loader: Optional[DataLoader],
                 config: TrainerConfig,
                 criterion: nn.Module,
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 logger: Optional[logging.Logger] = None):
        """初始化训练器"""
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.criterion = criterion
        
        # 设备
        self.device = torch.device(config.device)
        self.model.to(self.device)
        
        # 优化器
        if optimizer is None:
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        else:
            self.optimizer = optimizer
        
        # 调度器
        if scheduler is None:
            if config.scheduler_type == "cosine":
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, **config.scheduler_params
                )
            elif config.scheduler_type == "step":
                self.scheduler = optim.lr_scheduler.StepLR(
                    self.optimizer, **config.scheduler_params
                )
            elif config.scheduler_type == "plateau":
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, **config.scheduler_params
                )
            else:
                self.scheduler = None
        else:
            self.scheduler = scheduler
        
        # 日志
        self.logger = logger or self._setup_logger()
        self.writer = SummaryWriter(log_dir=config.log_dir)
        
        # 训练状态
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.global_step = 0
        self.current_epoch = 0
        
        # 检查点目录
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        
        # 混合精度训练
        self.scaler = torch.cuda.amp.GradScaler() if config.use_amp else None
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志器"""
        logger = logging.getLogger("hmm_trainer")
        logger.setLevel(logging.INFO)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # 文件处理器
        os.makedirs(self.config.log_dir, exist_ok=True)
        file_handler = logging.FileHandler(
            os.path.join(self.config.log_dir, "training.log")
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def train(self) -> Dict[str, Any]:
        """训练模型"""
        self.logger.info(f"Starting training for {self.config.epochs} epochs")
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.config.epochs):
            self.current_epoch = epoch
            self.logger.info(f"Epoch {epoch + 1}/{self.config.epochs}")
            
            # 训练一个epoch
            train_metrics = self.train_epoch()
            
            # 验证
            if self.val_loader is not None and (epoch + 1) % self.config.eval_every == 0:
                val_metrics = self.validate_epoch()
                
                # 保存最佳模型
                if val_metrics["val_loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["val_loss"]
                    self.patience_counter = 0
                    self.save_checkpoint(name="best")
                    self.logger.info(f"New best validation loss: {self.best_val_loss:.4f}")
                else:
                    self.patience_counter += 1
                
                # 早停
                if self.config.early_stopping and self.patience_counter >= self.config.patience:
                    self.logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
            
            # 调度器步进
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    val_loss = val_metrics["val_loss"] if "val_loss" in locals() else train_metrics["train_loss"]
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # 定期保存检查点
            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(name=f"epoch_{epoch + 1}")
        
        # 训练结束
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.2f} seconds")
        self.writer.close()
        
        return {"best_val_loss": self.best_val_loss}
    
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        epoch_loss = 0.0
        num_batches = len(self.train_loader)
        
        with tqdm(total=num_batches, desc=f"Epoch {self.current_epoch + 1} (train)") as pbar:
            for batch_idx, batch in enumerate(self.train_loader):
                # 准备数据
                batch = self._prepare_batch(batch)
                
                # 前向传播和损失计算
                if self.config.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(**batch)
                        loss = self.criterion(outputs)
                else:
                    outputs = self.model(**batch)
                    loss = self.criterion(outputs)
                
                # 反向传播
                self.optimizer.zero_grad()
                
                if self.config.use_amp:
                    self.scaler.scale(loss).backward()
                    
                    # 梯度裁剪
                    if self.config.grad_clip:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.grad_clip_norm
                        )
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    
                    # 梯度裁剪
                    if self.config.grad_clip:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.grad_clip_norm
                        )
                    
                    self.optimizer.step()
                
                # 更新统计信息
                epoch_loss += loss.item()
                
                # 记录日志
                if (batch_idx + 1) % self.config.log_every == 0:
                    self.writer.add_scalar("train/loss", loss.item(), self.global_step)
                    self.writer.add_scalar(
                        "train/lr", self.optimizer.param_groups[0]["lr"], self.global_step
                    )
                
                self.global_step += 1
                
                # 更新进度条
                pbar.update(1)
                pbar.set_postfix({"loss": loss.item()})
        
        epoch_loss /= num_batches
        self.logger.info(f"Epoch {self.current_epoch + 1} train loss: {epoch_loss:.4f}")
        
        return {"train_loss": epoch_loss}
    
    def validate_epoch(self) -> Dict[str, float]:
        """验证一个epoch"""
        self.model.eval()
        epoch_loss = 0.0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            with tqdm(total=num_batches, desc=f"Epoch {self.current_epoch + 1} (val)") as pbar:
                for batch in self.val_loader:
                    # 准备数据
                    batch = self._prepare_batch(batch)
                    
                    # 前向传播和损失计算
                    outputs = self.model(**batch)
                    loss = self.criterion(outputs)
                    
                    # 更新统计信息
                    epoch_loss += loss.item()
                    
                    # 更新进度条
                    pbar.update(1)
                    pbar.set_postfix({"loss": loss.item()})
        
        epoch_loss /= num_batches
        self.logger.info(f"Epoch {self.current_epoch + 1} val loss: {epoch_loss:.4f}")
        
        # 记录日志
        self.writer.add_scalar("val/loss", epoch_loss, self.current_epoch)
        
        return {"val_loss": epoch_loss}
    
    def save_checkpoint(self, name: str = "checkpoint") -> str:
        """保存检查点"""
        checkpoint_path = os.path.join(self.config.checkpoint_dir, f"{name}.pth")
        torch.save(
            {
                "epoch": self.current_epoch,
                "global_step": self.global_step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
                "best_val_loss": self.best_val_loss,
                "config": self.config,
            },
            checkpoint_path,
        )
        self.logger.info(f"Checkpoint saved to {checkpoint_path}")
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """加载检查点"""
        self.logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler and checkpoint["scheduler_state_dict"]:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        self.current_epoch = checkpoint["epoch"] + 1
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]
        
        self.logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    def _prepare_batch(self, batch: Union[Dict[str, torch.Tensor], Tuple]) -> Dict[str, torch.Tensor]:
        """准备批次数据"""
        if isinstance(batch, tuple):
            # 假设批次是LSS格式的元组：(imgs, rots, trans, intrins, post_rots, post_trans, binimgs)
            imgs, rots, trans, intrins, post_rots, post_trans, binimgs = batch
            batch_dict = {
                "lift_features": imgs.to(self.device),
                "splat_features": binimgs.to(self.device),
                "actions": torch.cat([rots, trans], dim=-1).to(self.device)
            }
            
            # 添加可选的深度特征
            if hasattr(self.model, "depth_features") and self.model.depth_features:
                batch_dict["depth_features"] = torch.zeros_like(imgs).to(self.device)
        
        elif isinstance(batch, dict):
            # 字典格式的批次
            batch_dict = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
        
        else:
            raise ValueError(f"Unsupported batch type: {type(batch)}")
        
        return batch_dict 