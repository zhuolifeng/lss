"""
HMM插件评估脚本
用于评估训练好的HMM插件在LSS模型上的性能

支持的评估指标：
1. IoU (Intersection over Union)
2. 时序一致性指标
3. 鲁棒性指标
4. 推理速度对比
"""

import argparse
import os
import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

# 导入LSS相关模块
import sys
sys.path.append('./src')
from models import get_model
from data import get_data_loaders
from tools import calculate_iou, get_val_info

# 导入HMM相关模块
from hmm_plugin import HMMPlugin, HMMConfig
from hmm_plugin.adapters import create_plug_and_play_hmm, global_hmm_manager
from train_hmm import FrozenLSSWithHMM


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate HMM Plugin Performance')
    parser.add_argument('--config', type=str, required=True, help='Path to LSS config file')
    parser.add_argument('--hmm_config', type=str, required=True, help='Path to HMM config file')
    parser.add_argument('--lss_checkpoint', type=str, required=True, help='Path to pre-trained LSS checkpoint')
    parser.add_argument('--hmm_checkpoint', type=str, required=True, help='Path to trained HMM checkpoint')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--save_visualizations', action='store_true', help='Save visualization results')
    parser.add_argument('--temporal_eval', action='store_true', help='Enable temporal consistency evaluation')
    return parser.parse_args()


class HMMEvaluator:
    """HMM插件评估器"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        self.output_dir = args.output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 加载模型
        self.lss_model, self.hmm_model = self._load_models()
        
        # 加载数据
        self.val_loader = self._load_data()
        
        # 评估指标
        self.metrics = {
            'iou_scores': [],
            'temporal_consistency': [],
            'inference_times': [],
            'baseline_ious': [],
            'enhanced_ious': []
        }
    
    def _load_models(self):
        """加载LSS和HMM模型"""
        # 加载LSS配置
        with open(self.args.config, 'r') as f:
            lss_config = json.load(f)
        
        # 加载HMM配置
        with open(self.args.hmm_config, 'r') as f:
            hmm_config_dict = json.load(f)
        
        # 创建LSS模型
        lss_model = get_model(lss_config)
        lss_checkpoint = torch.load(self.args.lss_checkpoint, map_location=self.device)
        lss_model.load_state_dict(lss_checkpoint['model_state_dict'])
        lss_model.to(self.device)
        lss_model.eval()
        
        # 创建HMM插件
        hmm_config = HMMConfig(**hmm_config_dict)
        hmm_plugin = HMMPlugin(hmm_config)
        
        # 创建包装器
        hmm_model = FrozenLSSWithHMM(lss_model, hmm_plugin)
        
        # 加载HMM检查点
        hmm_checkpoint = torch.load(self.args.hmm_checkpoint, map_location=self.device)
        hmm_model.hmm_plugin.load_state_dict(hmm_checkpoint['hmm_plugin_state_dict'])
        hmm_model.to(self.device)
        hmm_model.eval()
        
        print(f"模型加载完成，移动到设备: {self.device}")
        print(f"HMM插件参数数量: {sum(p.numel() for p in hmm_model.hmm_plugin.parameters())}")
        
        return lss_model, hmm_model
    
    def _load_data(self):
        """加载验证数据"""
        with open(self.args.config, 'r') as f:
            config = json.load(f)
        
        _, val_loader = get_data_loaders(config, self.args.batch_size)
        print(f"验证数据加载完成，批次数量: {len(val_loader)}")
        return val_loader
    
    def evaluate_performance(self):
        """评估性能指标"""
        print("\n开始性能评估...")
        
        total_samples = 0
        total_iou_baseline = 0
        total_iou_enhanced = 0
        total_inference_time = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="评估中")):
                imgs, rots, trans, intrins, post_rots, post_trans, binimgs = batch
                
                # 移动到设备
                imgs = imgs.to(self.device)
                rots = rots.to(self.device)
                trans = trans.to(self.device)
                intrins = intrins.to(self.device)
                post_rots = post_rots.to(self.device)
                post_trans = post_trans.to(self.device)
                binimgs = binimgs.to(self.device)
                
                # 测量推理时间
                start_time = time.time()
                
                # 前向传播
                outputs = self.hmm_model(imgs, rots, trans, intrins, post_rots, post_trans, binimgs)
                
                inference_time = time.time() - start_time
                total_inference_time += inference_time
                
                # 计算IoU
                baseline_iou = calculate_iou(outputs['lss_baseline_output'], binimgs)
                enhanced_iou = calculate_iou(outputs['enhanced_output'], binimgs)
                
                # 累计统计
                batch_size = imgs.shape[0]
                total_samples += batch_size
                total_iou_baseline += baseline_iou * batch_size
                total_iou_enhanced += enhanced_iou * batch_size
                
                # 保存批次结果
                self.metrics['baseline_ious'].append(baseline_iou)
                self.metrics['enhanced_ious'].append(enhanced_iou)
                self.metrics['inference_times'].append(inference_time / batch_size)
                
                # 时序一致性评估
                if self.args.temporal_eval and batch_idx > 0:
                    temporal_score = self._calculate_temporal_consistency(outputs, batch_idx)
                    self.metrics['temporal_consistency'].append(temporal_score)
        
        # 计算平均指标
        avg_baseline_iou = total_iou_baseline / total_samples
        avg_enhanced_iou = total_iou_enhanced / total_samples
        avg_inference_time = total_inference_time / total_samples
        
        # 计算提升
        iou_improvement = avg_enhanced_iou - avg_baseline_iou
        improvement_percentage = (iou_improvement / avg_baseline_iou) * 100
        
        # 打印结果
        print(f"\n=== 评估结果 ===")
        print(f"基线LSS IoU: {avg_baseline_iou:.4f}")
        print(f"HMM增强IoU: {avg_enhanced_iou:.4f}")
        print(f"IoU提升: {iou_improvement:.4f} ({improvement_percentage:.2f}%)")
        print(f"平均推理时间: {avg_inference_time:.4f}s")
        
        if self.args.temporal_eval and self.metrics['temporal_consistency']:
            avg_temporal = np.mean(self.metrics['temporal_consistency'])
            print(f"时序一致性得分: {avg_temporal:.4f}")
        
        # 保存结果
        self._save_results({
            'avg_baseline_iou': avg_baseline_iou,
            'avg_enhanced_iou': avg_enhanced_iou,
            'iou_improvement': iou_improvement,
            'improvement_percentage': improvement_percentage,
            'avg_inference_time': avg_inference_time,
            'total_samples': total_samples
        })
        
        return {
            'baseline_iou': avg_baseline_iou,
            'enhanced_iou': avg_enhanced_iou,
            'improvement': iou_improvement,
            'inference_time': avg_inference_time
        }
    
    def _calculate_temporal_consistency(self, outputs, batch_idx):
        """计算时序一致性分数"""
        # 简化的时序一致性计算
        # 实际应用中需要更复杂的时序建模
        if not hasattr(self, 'prev_features'):
            self.prev_features = outputs['extracted_features']['splat_features']
            return 1.0
        
        current_features = outputs['extracted_features']['splat_features']
        
        # 计算特征相似度
        cosine_sim = torch.cosine_similarity(
            current_features.flatten(1), 
            self.prev_features.flatten(1),
            dim=1
        ).mean().item()
        
        self.prev_features = current_features
        return cosine_sim
    
    def _save_results(self, results):
        """保存评估结果"""
        # 保存JSON结果
        with open(os.path.join(self.output_dir, 'evaluation_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        # 保存详细指标
        np.savez(
            os.path.join(self.output_dir, 'detailed_metrics.npz'),
            baseline_ious=self.metrics['baseline_ious'],
            enhanced_ious=self.metrics['enhanced_ious'],
            inference_times=self.metrics['inference_times'],
            temporal_consistency=self.metrics['temporal_consistency']
        )
        
        print(f"评估结果已保存到: {self.output_dir}")
    
    def create_visualizations(self):
        """创建可视化结果"""
        if not self.args.save_visualizations:
            return
        
        print("\n创建可视化结果...")
        
        # 设置图表样式
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('HMM插件性能评估', fontsize=16)
        
        # 1. IoU对比
        axes[0, 0].hist(self.metrics['baseline_ious'], alpha=0.7, label='基线LSS', bins=30)
        axes[0, 0].hist(self.metrics['enhanced_ious'], alpha=0.7, label='HMM增强', bins=30)
        axes[0, 0].set_xlabel('IoU Score')
        axes[0, 0].set_ylabel('频次')
        axes[0, 0].set_title('IoU分布对比')
        axes[0, 0].legend()
        
        # 2. 推理时间分布
        axes[0, 1].hist(self.metrics['inference_times'], bins=30, alpha=0.7, color='orange')
        axes[0, 1].set_xlabel('推理时间 (秒)')
        axes[0, 1].set_ylabel('频次')
        axes[0, 1].set_title('推理时间分布')
        
        # 3. IoU改善散点图
        baseline_array = np.array(self.metrics['baseline_ious'])
        enhanced_array = np.array(self.metrics['enhanced_ious'])
        improvement = enhanced_array - baseline_array
        
        axes[1, 0].scatter(baseline_array, improvement, alpha=0.6, s=20)
        axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        axes[1, 0].set_xlabel('基线IoU')
        axes[1, 0].set_ylabel('IoU改善')
        axes[1, 0].set_title('IoU改善 vs 基线性能')
        
        # 4. 时序一致性（如果有）
        if self.metrics['temporal_consistency']:
            axes[1, 1].plot(self.metrics['temporal_consistency'], alpha=0.7)
            axes[1, 1].set_xlabel('批次索引')
            axes[1, 1].set_ylabel('时序一致性得分')
            axes[1, 1].set_title('时序一致性变化')
        else:
            axes[1, 1].text(0.5, 0.5, '未启用时序评估', ha='center', va='center', 
                          transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('时序一致性（未启用）')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'evaluation_plots.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"可视化结果已保存到: {self.output_dir}/evaluation_plots.png")
    
    def run_evaluation(self):
        """运行完整评估"""
        print("开始HMM插件评估...")
        
        # 性能评估
        performance_results = self.evaluate_performance()
        
        # 创建可视化
        self.create_visualizations()
        
        # 生成报告
        self._generate_report(performance_results)
        
        return performance_results
    
    def _generate_report(self, results):
        """生成评估报告"""
        report = f"""
# HMM插件评估报告

## 基本信息
- 评估时间: {time.strftime('%Y-%m-%d %H:%M:%S')}
- 设备: {self.device}
- 批次大小: {self.args.batch_size}
- 总样本数: {len(self.val_loader) * self.args.batch_size}

## 性能指标
- 基线LSS IoU: {results['baseline_iou']:.4f}
- HMM增强IoU: {results['enhanced_iou']:.4f}
- IoU提升: {results['improvement']:.4f} ({(results['improvement']/results['baseline_iou']*100):.2f}%)
- 平均推理时间: {results['inference_time']:.4f}s

## 结论
{'HMM插件显著提升了LSS模型的分割性能' if results['improvement'] > 0.01 else 'HMM插件对LSS模型的改善有限'}

## 建议
{'建议在实际部署中使用HMM插件' if results['improvement'] > 0.01 else '建议进一步优化HMM插件参数'}
        """
        
        with open(os.path.join(self.output_dir, 'evaluation_report.md'), 'w') as f:
            f.write(report)
        
        print(f"评估报告已保存到: {self.output_dir}/evaluation_report.md")


def main():
    args = parse_args()
    
    # 创建评估器
    evaluator = HMMEvaluator(args)
    
    # 运行评估
    results = evaluator.run_evaluation()
    
    print("\n评估完成！")
    return results


if __name__ == "__main__":
    main()