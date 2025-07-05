"""
修正的HMM插件训练脚本
训练策略：冻结LSS模型参数，只训练HMM插件

核心思想：
1. LSS模型用于评估过程（提供特征和最终结果）
2. HMM插件独立训练，不影响LSS模型权重
3. 支持即插即用和一次训练多次使用
"""

import argparse
import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import random
from typing import Dict, Optional, List, Tuple

# 导入LSS相关模块
sys.path.append('./src')
from models import get_model  # 导入真实的LSS模型
from data import get_data_loaders  # 导入数据加载器
from tools import get_val_info, calculate_iou  # 导入评估工具

# 导入HMM相关模块
from hmm_plugin import HMMPlugin, HMMConfig
from hmm_plugin.adapters import (
    create_plug_and_play_hmm, 
    register_shared_hmm_plugin,
    global_hmm_manager
)
from hmm_plugin.training.loss import create_hmm_loss


def parse_args():
    parser = argparse.ArgumentParser(description='Train HMM Plugin with Frozen LSS Model')
    parser.add_argument('--config', type=str, required=True, help='Path to LSS config file')
    parser.add_argument('--hmm_config', type=str, required=True, help='Path to HMM config file')
    parser.add_argument('--lss_checkpoint', type=str, required=True, help='Path to pre-trained LSS checkpoint')
    parser.add_argument('--output_dir', type=str, default='./hmm_experiments', help='Output directory')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--save_every', type=int, default=10, help='Save checkpoint every N epochs')
    parser.add_argument('--eval_every', type=int, default=5, help='Evaluate every N epochs')
    return parser.parse_args()


class FrozenLSSWithHMM(nn.Module):
    """
    冻结LSS模型，只训练HMM插件的包装器
    
    关键设计：
    1. LSS模型参数完全冻结
    2. HMM插件独立训练
    3. 在评估时通过完整管道计算最终损失
    """
    
    def __init__(self, lss_model: nn.Module, hmm_plugin: HMMPlugin):
        super().__init__()
        
        # 冻结LSS模型
        self.lss_model = lss_model
        for param in self.lss_model.parameters():
            param.requires_grad = False
        
        # HMM插件（可训练）
        self.hmm_plugin = hmm_plugin
        
        # 特征提取钩子
        self.extracted_features = {}
        self._register_hooks()
        
    def _register_hooks(self):
        """注册钩子来提取LSS中间特征"""
        
        def extract_lift_features(module, input, output):
            self.extracted_features['lift_features'] = output.detach()
        
        def extract_splat_features(module, input, output):
            self.extracted_features['splat_features'] = output.detach()
        
        def extract_depth_features(module, input, output):
            self.extracted_features['depth_features'] = output.detach()
        
        # 根据LSS模型结构注册钩子
        for name, module in self.lss_model.named_modules():
            if any(keyword in name.lower() for keyword in ['lift', 'get_cam_feats']):
                module.register_forward_hook(extract_lift_features)
            elif any(keyword in name.lower() for keyword in ['splat', 'bev_pool']):
                module.register_forward_hook(extract_splat_features)
            elif 'depth' in name.lower():
                module.register_forward_hook(extract_depth_features)
    
    def forward(self, imgs, rots, trans, intrins, post_rots, post_trans, binimgs):
        """
        前向传播策略：
        1. 使用冻结的LSS模型提取特征和计算基线结果
        2. 使用HMM插件增强特征
        3. 通过增强特征重新计算最终结果（用于评估）
        """
        
        # 1. LSS基线前向传播（冻结参数）
        with torch.no_grad():
            lss_baseline_output = self.lss_model(imgs, rots, trans, intrins, post_rots, post_trans)
        
        # 2. 提取LSS中间特征
        lift_features = self.extracted_features.get('lift_features')
        splat_features = self.extracted_features.get('splat_features') 
        depth_features = self.extracted_features.get('depth_features')
        
        # 3. 提取动作信息
        actions = self._extract_actions(rots, trans)
        
        # 4. HMM插件处理（可训练）
        hmm_outputs = self.hmm_plugin(
            lift_features=lift_features,
            splat_features=splat_features,
            actions=actions,
            depth_features=depth_features,
            training=self.training
        )
        
        # 5. 使用增强特征重新计算最终输出（用于评估HMM效果）
        enhanced_output = self._compute_enhanced_output(
            hmm_outputs['enhanced_splat'], 
            rots, trans, intrins, post_rots, post_trans
        )
        
        return {
            'lss_baseline_output': lss_baseline_output,
            'enhanced_output': enhanced_output,
            'hmm_outputs': hmm_outputs,
            'extracted_features': {
                'lift_features': lift_features,
                'splat_features': splat_features,
                'depth_features': depth_features
            },
            'actions': actions
        }
    
    def _extract_actions(self, rots, trans):
        """从LSS输入中提取动作信息"""
        batch_size = rots.shape[0]
        
        # 简化的动作提取（实际应用中需要更复杂的计算）
        if rots.dim() >= 3:
            # 计算相邻帧之间的旋转和平移差异
            rot_diff = rots[:, 1:] - rots[:, :-1] if rots.shape[1] > 1 else torch.zeros_like(rots[:, :1])
            trans_diff = trans[:, 1:] - trans[:, :-1] if trans.shape[1] > 1 else torch.zeros_like(trans[:, :1])
            
            # 组合成动作向量 [batch_size, 6] (3D旋转 + 3D平移)
            actions = torch.cat([
                rot_diff.view(batch_size, -1)[:, :3],
                trans_diff.view(batch_size, -1)[:, :3]
            ], dim=-1)
        else:
            actions = torch.zeros(batch_size, 6, device=rots.device)
        
        return actions
    
    def _compute_enhanced_output(self, enhanced_splat, rots, trans, intrins, post_rots, post_trans):
        """使用增强的splat特征计算最终输出"""
        # 这里需要根据具体的LSS模型实现
        # 通常是将增强的splat特征通过最终的分割头处理
        
        # 简化实现：假设LSS模型有一个final_layer
        if hasattr(self.lss_model, 'final_layer'):
            with torch.enable_grad():  # 允许梯度传播到HMM插件
                enhanced_output = self.lss_model.final_layer(enhanced_splat)
        else:
            # 如果没有分离的final_layer，使用简单的线性层
            if not hasattr(self, '_temp_final_layer'):
                self._temp_final_layer = nn.Linear(
                    enhanced_splat.shape[-1], 200  # 假设200个分割类别
                ).to(enhanced_splat.device)
            enhanced_output = self._temp_final_layer(enhanced_splat)
        
        return enhanced_output


def create_training_setup(args):
    """创建训练设置"""
    
    # 1. 加载配置
    with open(args.config, 'r') as f:
        lss_config = json.load(f)
    
    with open(args.hmm_config, 'r') as f:
        hmm_config_dict = json.load(f)
    
    hmm_config = HMMConfig(**hmm_config_dict)
    
    # 2. 创建LSS模型并加载预训练权重
    lss_model = get_model(lss_config)
    checkpoint = torch.load(args.lss_checkpoint, map_location='cpu')
    lss_model.load_state_dict(checkpoint['model_state_dict'])
    
    # 3. 创建HMM插件
    hmm_plugin = HMMPlugin(hmm_config)
    
    # 4. 创建包装模型
    model = FrozenLSSWithHMM(lss_model, hmm_plugin)
    
    # 5. 创建数据加载器
    train_loader, val_loader = get_data_loaders(lss_config)
    
    # 6. 创建优化器（只优化HMM插件参数）
    hmm_params = list(hmm_plugin.parameters())
    optimizer = optim.AdamW(hmm_params, lr=args.lr, weight_decay=1e-5)
    
    # 7. 创建损失函数
    hmm_criterion = create_hmm_loss(hmm_config_dict)
    lss_criterion = nn.CrossEntropyLoss()  # 用于评估最终结果
    
    # 8. 创建学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    return model, train_loader, val_loader, optimizer, scheduler, hmm_criterion, lss_criterion


def train_epoch(model, train_loader, optimizer, hmm_criterion, lss_criterion, device, epoch, writer):
    """训练一个epoch"""
    
    model.train()
    model.hmm_plugin.train()  # 确保HMM插件在训练模式
    
    total_loss = 0
    total_hmm_loss = 0
    total_lss_loss = 0
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, batch in enumerate(progress_bar):
        # 移动数据到设备
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        
        # 前向传播
        outputs = model(
            batch['imgs'], batch['rots'], batch['trans'], 
            batch['intrins'], batch['post_rots'], batch['post_trans'], 
            batch['binimgs']
        )
        
        # 计算HMM损失（这是主要训练目标）
        hmm_losses = hmm_criterion(
            # 深度似然 P(Dt|Xt)
            outputs['extracted_features']['lift_features'][:, :-1],  # Xt
            outputs['extracted_features']['depth_features'][:, :-1] if outputs['extracted_features']['depth_features'] is not None else torch.zeros_like(outputs['extracted_features']['lift_features'][:, :-1]),  # Dt
            
            # 测量似然 P(Mt|Yt)
            outputs['extracted_features']['splat_features'][:, :-1],  # Yt
            batch['binimgs'][:, :-1],  # Mt (测量/标签)
            
            # BEV特征KL散度数据
            outputs['extracted_features']['lift_features'][:, 1:],   # Xt+1
            outputs['extracted_features']['splat_features'][:, :-1], # Yt
            outputs['actions'][:, :-1] if outputs['actions'].shape[1] > 1 else outputs['actions'],  # At
            outputs['extracted_features']['splat_features'][:, 1:],  # Yt+1_true
            
            # 状态转移KL散度数据
            batch['imgs'][:, :-1],  # Ot (观测)
            outputs['extracted_features']['lift_features'][:, :-1],  # Xt
            outputs['extracted_features']['lift_features'][:, 1:]    # Xt+1_true
        )
        
        # 计算LSS损失（用于评估效果）
        with torch.no_grad():
            lss_loss = lss_criterion(
                outputs['enhanced_output'].view(-1, outputs['enhanced_output'].shape[-1]),
                batch['binimgs'].view(-1).long()
            )
        
        # 总损失（只有HMM损失参与反向传播）
        total_batch_loss = hmm_losses['total_loss']
        
        # 反向传播（只更新HMM参数）
        optimizer.zero_grad()
        total_batch_loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.hmm_plugin.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # 记录损失
        total_loss += total_batch_loss.item()
        total_hmm_loss += hmm_losses['total_loss'].item()
        total_lss_loss += lss_loss.item()
        
        # 更新进度条
        progress_bar.set_postfix({
            'HMM Loss': f'{hmm_losses["total_loss"].item():.4f}',
            'LSS Loss': f'{lss_loss.item():.4f}',
            'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })
        
        # 记录到tensorboard
        if batch_idx % 10 == 0:
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Train/HMM_Loss', hmm_losses['total_loss'].item(), global_step)
            writer.add_scalar('Train/LSS_Loss', lss_loss.item(), global_step)
            writer.add_scalar('Train/Learning_Rate', optimizer.param_groups[0]['lr'], global_step)
            
            # 记录各个HMM损失组件
            for loss_name, loss_value in hmm_losses.items():
                if 'loss' in loss_name and loss_name != 'total_loss':
                    writer.add_scalar(f'Train/HMM_{loss_name}', loss_value.item(), global_step)
    
    avg_loss = total_loss / len(train_loader)
    avg_hmm_loss = total_hmm_loss / len(train_loader)
    avg_lss_loss = total_lss_loss / len(train_loader)
    
    return {
        'total_loss': avg_loss,
        'hmm_loss': avg_hmm_loss,
        'lss_loss': avg_lss_loss
    }


def validate_epoch(model, val_loader, hmm_criterion, lss_criterion, device, epoch, writer):
    """验证一个epoch"""
    
    model.eval()
    
    total_loss = 0
    total_hmm_loss = 0
    total_lss_loss = 0
    total_iou = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc='Validation')):
            # 移动数据到设备
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            
            # 前向传播
            outputs = model(
                batch['imgs'], batch['rots'], batch['trans'], 
                batch['intrins'], batch['post_rots'], batch['post_trans'], 
                batch['binimgs']
            )
            
            # 计算损失（验证时也计算HMM损失）
            hmm_losses = hmm_criterion(
                outputs['extracted_features']['lift_features'][:, :-1],
                outputs['extracted_features']['depth_features'][:, :-1] if outputs['extracted_features']['depth_features'] is not None else torch.zeros_like(outputs['extracted_features']['lift_features'][:, :-1]),
                outputs['extracted_features']['splat_features'][:, :-1],
                batch['binimgs'][:, :-1],
                outputs['extracted_features']['lift_features'][:, 1:],
                outputs['extracted_features']['splat_features'][:, :-1],
                outputs['actions'][:, :-1] if outputs['actions'].shape[1] > 1 else outputs['actions'],
                outputs['extracted_features']['splat_features'][:, 1:],
                batch['imgs'][:, :-1],
                outputs['extracted_features']['lift_features'][:, :-1],
                outputs['extracted_features']['lift_features'][:, 1:]
            )
            
            lss_loss = lss_criterion(
                outputs['enhanced_output'].view(-1, outputs['enhanced_output'].shape[-1]),
                batch['binimgs'].view(-1).long()
            )
            
            # 计算IoU（评估最终效果）
            pred = outputs['enhanced_output'].argmax(dim=-1)
            target = batch['binimgs']
            iou = calculate_iou(pred, target)
            
            total_loss += hmm_losses['total_loss'].item()
            total_hmm_loss += hmm_losses['total_loss'].item()
            total_lss_loss += lss_loss.item()
            total_iou += iou
    
    avg_loss = total_loss / len(val_loader)
    avg_hmm_loss = total_hmm_loss / len(val_loader)
    avg_lss_loss = total_lss_loss / len(val_loader)
    avg_iou = total_iou / len(val_loader)
    
    # 记录验证结果
    writer.add_scalar('Val/HMM_Loss', avg_hmm_loss, epoch)
    writer.add_scalar('Val/LSS_Loss', avg_lss_loss, epoch)
    writer.add_scalar('Val/IoU', avg_iou, epoch)
    
    return {
        'total_loss': avg_loss,
        'hmm_loss': avg_hmm_loss,
        'lss_loss': avg_lss_loss,
        'iou': avg_iou
    }


def main():
    args = parse_args()
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置tensorboard
    writer = SummaryWriter(os.path.join(args.output_dir, 'tensorboard'))
    
    # 创建训练设置
    model, train_loader, val_loader, optimizer, scheduler, hmm_criterion, lss_criterion = create_training_setup(args)
    model.to(device)
    
    print(f"Model moved to {device}")
    print(f"HMM Plugin parameters: {sum(p.numel() for p in model.hmm_plugin.parameters() if p.requires_grad)}")
    print(f"LSS Model parameters (frozen): {sum(p.numel() for p in model.lss_model.parameters())}")
    
    # 训练循环
    best_iou = 0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # 训练
        train_stats = train_epoch(
            model, train_loader, optimizer, hmm_criterion, lss_criterion, 
            device, epoch, writer
        )
        
        # 验证
        if epoch % args.eval_every == 0:
            val_stats = validate_epoch(
                model, val_loader, hmm_criterion, lss_criterion, 
                device, epoch, writer
            )
            
            print(f"Train - HMM Loss: {train_stats['hmm_loss']:.4f}, LSS Loss: {train_stats['lss_loss']:.4f}")
            print(f"Val - HMM Loss: {val_stats['hmm_loss']:.4f}, LSS Loss: {val_stats['lss_loss']:.4f}, IoU: {val_stats['iou']:.4f}")
            
            # 保存最佳模型
            if val_stats['iou'] > best_iou:
                best_iou = val_stats['iou']
                torch.save({
                    'epoch': epoch,
                    'hmm_plugin_state_dict': model.hmm_plugin.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_iou': best_iou,
                    'hmm_config': model.hmm_plugin.config,
                }, os.path.join(args.output_dir, 'best_hmm_plugin.pth'))
        
        # 保存定期检查点
        if epoch % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'hmm_plugin_state_dict': model.hmm_plugin.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, os.path.join(args.output_dir, f'hmm_plugin_epoch_{epoch}.pth'))
        
        # 更新学习率
        scheduler.step()
    
    # 训练完成后注册到全局管理器
    register_shared_hmm_plugin(
        plugin_name="trained_hmm_general",
        hmm_plugin=model.hmm_plugin,
        config=model.hmm_plugin.get_state_info()
    )
    
    print(f"\nTraining completed! Best IoU: {best_iou:.4f}")
    print("HMM Plugin registered to global manager for reuse.")
    
    writer.close()


if __name__ == '__main__':
    main() 