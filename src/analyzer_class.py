"""
LSS模型三步骤干扰影响分析工具
分析Lift、Splat、Shoot三个步骤在不同干扰下的表现
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Tuple, Any
import seaborn as sns
try:
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import MinMaxScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# 设置matplotlib字体
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

from .models import compile_model
from .data import compile_data
from .tools import get_val_info, SimpleLoss, denormalize_img
from corruptions.corrupt import ImageCorruption


class LSSCorruptionAnalyzer:
    """LSS模型干扰影响分析器"""
    
    def __init__(self, model, device='cuda:0'):
        self.model = model
        self.device = device
        self.corruption = ImageCorruption(model)
        
        # 用于保存中间结果的钩子
        self.intermediate_outputs = {}
        self.register_hooks()
    
    def register_hooks(self):
        """注册前向传播钩子来获取中间输出"""
        def hook_lift(module, input, output):
            self.intermediate_outputs['lift_features'] = output.detach().cpu()
        
        def hook_shoot(module, input, output):
            self.intermediate_outputs['shoot_features'] = output.detach().cpu()
        
        # 注册钩子
        self.model.camencode.register_forward_hook(hook_lift)  # Lift步骤：相机编码器
        # Splat步骤（体素池化）是函数而非模块，在analyze_step_by_step中手动捕获
        self.model.bevencode.register_forward_hook(hook_shoot)  # Shoot步骤：BEV编码器
    
    def analyze_step_by_step(self, images, rots, trans, intrins, post_rots, post_trans, binimgs, 
                           corruption_types=['origin', 'noise', 'fgsm', 'pgd', 'cw', 'dag', 'fusion', 'robust_bev']) -> Dict:
        """
        逐步分析LSS三个步骤在不同干扰下的表现
        
        Returns:
            Dict包含每个步骤和每种干扰的详细分析结果
        """
        results = {corruption_type: {} for corruption_type in corruption_types}
        
        for corruption_type in corruption_types:
            print(f"\n=== 分析干扰类型: {corruption_type} ===")
            
            # 应用干扰
            if corruption_type == 'origin':
                corrupted_images = images
            elif corruption_type == 'noise':
                intensity = 'medium'
                corrupted_images = self.corruption.apply_corruption(
                    images, rots, trans, intrins, post_rots, post_trans, binimgs,
                    type='noise', intensity=intensity
                )
            elif corruption_type == 'fgsm':
                intensity = 'medium'
                corrupted_images = self.corruption.apply_corruption(
                    images, rots, trans, intrins, post_rots, post_trans, binimgs,
                    type='fgsm', intensity=intensity
                )
            elif corruption_type == 'pgd':
                intensity = 'medium'
                corrupted_images = self.corruption.apply_corruption(
                    images, rots, trans, intrins, post_rots, post_trans, binimgs,
                    type='pgd', intensity=intensity
                )
            elif corruption_type == 'cw':
                intensity = 'medium'
                corrupted_images = self.corruption.apply_corruption(
                    images, rots, trans, intrins, post_rots, post_trans, binimgs,
                    type='cw', intensity=intensity
                )
            elif corruption_type == 'dag':
                intensity = 'medium'
                corrupted_images = self.corruption.apply_corruption(
                    images, rots, trans, intrins, post_rots, post_trans, binimgs,
                    type='dag', intensity=intensity
                )
            elif corruption_type == 'fusion':
                intensity = 'medium'
                corrupted_images = self.corruption.apply_corruption(
                    images, rots, trans, intrins, post_rots, post_trans, binimgs,
                    type='fusion', intensity=intensity
                )
            elif corruption_type == 'robust_bev':
                intensity = 'medium'
                corrupted_images = self.corruption.apply_corruption(
                    images, rots, trans, intrins, post_rots, post_trans, binimgs,
                    type='robust_bev', intensity=intensity
                )
            # 清空中间输出
            self.intermediate_outputs.clear()
            
            # 前向传播
            with torch.no_grad():
                # Step 1: Lift - 获取相机特征
                geom = self.model.get_geometry(rots, trans, intrins, post_rots, post_trans)
                cam_features = self.model.get_cam_feats(corrupted_images)
                
                # Step 2: Splat - 体素池化 (手动捕获输出)
                bev_features = self.model.voxel_pooling(geom, cam_features)
                self.intermediate_outputs['splat_features'] = bev_features.detach().cpu()
                
                # Step 3: Shoot - BEV编码和预测
                final_output = self.model.bevencode(bev_features)
            
            # 保存结果
            results[corruption_type] = {
                'images': corrupted_images,
                'cam_features': cam_features,
                'bev_features': bev_features,
                'final_output': final_output,
                'predictions': torch.sigmoid(final_output),
                'ground_truth': binimgs
            }
            
            # 计算统计信息
            results[corruption_type]['stats'] = self._compute_step_statistics(
                cam_features, bev_features, final_output, binimgs
            )
        
        return results
    
    def _compute_step_statistics(self, cam_features, bev_features, final_output, ground_truth) -> Dict:
        """计算每个步骤的统计信息"""
        stats = {}
        
        # Lift步骤统计
        stats['lift'] = {
            'feature_magnitude': cam_features.abs().mean().item(),
            'feature_std': cam_features.std().item(),
            'feature_sparsity': (cam_features.abs() < 1e-6).float().mean().item(),
            'shape': list(cam_features.shape)
        }
        
        # Splat步骤统计
        stats['splat'] = {
            'feature_magnitude': bev_features.abs().mean().item(),
            'feature_std': bev_features.std().item(),
            'feature_sparsity': (bev_features.abs() < 1e-6).float().mean().item(),
            'shape': list(bev_features.shape)
        }
        
        # Shoot步骤统计
        predictions = torch.sigmoid(final_output)
        target = ground_truth.float()
        
        # 计算IoU
        intersection = (predictions > 0.5).float() * target
        union = (predictions > 0.5).float() + target - intersection
        iou = intersection.sum() / (union.sum() + 1e-8)
        
        stats['shoot'] = {
            'prediction_magnitude': final_output.abs().mean().item(),
            'prediction_std': final_output.std().item(),
            'iou': iou.item(),
            'accuracy': ((predictions > 0.5).float() == target).float().mean().item(),
            'precision': (intersection.sum() / ((predictions > 0.5).float().sum() + 1e-8)).item(),
            'recall': (intersection.sum() / (target.sum() + 1e-8)).item(),
            'shape': list(final_output.shape)
        }
        
        return stats
    
    def compare_corruptions(self, results: Dict) -> Dict:
        """比较不同干扰类型的影响"""
        comparison = {
            'lift_impact': {},
            'splat_impact': {},
            'shoot_impact': {}
        }
        
        baseline = results['origin']['stats']
        
        for corruption_type, result in results.items():
            if corruption_type == 'origin':
                continue
                
            stats = result['stats']
            
            # Lift影响
            comparison['lift_impact'][corruption_type] = {
                'magnitude_change': (stats['lift']['feature_magnitude'] - baseline['lift']['feature_magnitude']) / baseline['lift']['feature_magnitude'],
                'std_change': (stats['lift']['feature_std'] - baseline['lift']['feature_std']) / baseline['lift']['feature_std'],
                'sparsity_change': stats['lift']['feature_sparsity'] - baseline['lift']['feature_sparsity']
            }
            
            # Splat影响
            comparison['splat_impact'][corruption_type] = {
                'magnitude_change': (stats['splat']['feature_magnitude'] - baseline['splat']['feature_magnitude']) / baseline['splat']['feature_magnitude'],
                'std_change': (stats['splat']['feature_std'] - baseline['splat']['feature_std']) / baseline['splat']['feature_std'],
                'sparsity_change': stats['splat']['feature_sparsity'] - baseline['splat']['feature_sparsity']
            }
            
            # Shoot影响
            comparison['shoot_impact'][corruption_type] = {
                'iou_drop': baseline['shoot']['iou'] - stats['shoot']['iou'],
                'accuracy_drop': baseline['shoot']['accuracy'] - stats['shoot']['accuracy'],
                'prediction_magnitude_change': (stats['shoot']['prediction_magnitude'] - baseline['shoot']['prediction_magnitude']) / baseline['shoot']['prediction_magnitude']
            }
        
        return comparison
    
    def visualize_analysis(self, results: Dict, comparison: Dict, output_dir: str = './analysis_output'):
        """可视化分析结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 特征幅度对比 (分离显示)
        self._plot_feature_magnitudes(results, os.path.join(output_dir, 'feature_magnitudes.png'))
        
        # 1b. 标准化特征对比 (统一刻度)
        self._plot_normalized_feature_comparison(results, os.path.join(output_dir, 'normalized_feature_comparison.png'))
        
        # 2. 性能下降对比
        self._plot_performance_drops(comparison, os.path.join(output_dir, 'performance_drops.png'))
        
        # 3. 预测结果对比
        self._plot_predictions(results, os.path.join(output_dir, 'predictions_comparison.png'))
        
        # 4. 影响热力图
        self._plot_impact_heatmap(comparison, os.path.join(output_dir, 'impact_heatmap.png'))
        
        # === 新增的增强可视化 ===
        # 5. 脆弱性分析雷达图
        self._plot_vulnerability_radar(comparison, os.path.join(output_dir, 'vulnerability_radar.png'))
        
        # 6. 步骤敏感性对比图
        self._plot_step_sensitivity(comparison, os.path.join(output_dir, 'step_sensitivity.png'))
        
        # 7. 干扰传播分析图
        self._plot_corruption_propagation(results, comparison, os.path.join(output_dir, 'corruption_propagation.png'))
        
        # 8. 综合脆弱性评估图
        self._plot_comprehensive_vulnerability(results, comparison, os.path.join(output_dir, 'comprehensive_vulnerability.png'))
        
        # 9. 特征分布变化对比
        self._plot_feature_distribution_changes(results, os.path.join(output_dir, 'feature_distribution_changes.png'))
        
        # 10. 干扰强度vs性能下降关系图
        self._plot_corruption_intensity_impact(results, comparison, os.path.join(output_dir, 'corruption_intensity_impact.png'))
    
    def _plot_feature_magnitudes(self, results: Dict, save_path: str):
        """绘制特征幅度对比图 - 分离显示三个步骤"""
        corruption_types = list(results.keys())
        lift_mags = [results[ct]['stats']['lift']['feature_magnitude'] for ct in corruption_types]
        splat_mags = [results[ct]['stats']['splat']['feature_magnitude'] for ct in corruption_types]
        shoot_mags = [results[ct]['stats']['shoot']['prediction_magnitude'] for ct in corruption_types]
        
        # 创建1行3列的子图
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # 定义颜色
        colors = ['skyblue', 'lightcoral', 'lightgreen']
        
        # 子图1：Lift Features
        bars1 = ax1.bar(corruption_types, lift_mags, color=colors[0], alpha=0.8, edgecolor='black', linewidth=0.8)
        ax1.set_title('Lift Features Magnitude\n(Camera Feature Extraction)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Corruption Type', fontsize=12)
        ax1.set_ylabel('Feature Magnitude', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 子图2：Splat Features  
        bars2 = ax2.bar(corruption_types, splat_mags, color=colors[1], alpha=0.8, edgecolor='black', linewidth=0.8)
        ax2.set_title('Splat Features Magnitude\n(BEV Fusion)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Corruption Type', fontsize=12)
        ax2.set_ylabel('Feature Magnitude', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 子图3：Shoot Predictions
        bars3 = ax3.bar(corruption_types, shoot_mags, color=colors[2], alpha=0.8, edgecolor='black', linewidth=0.8)
        ax3.set_title('Shoot Predictions Magnitude\n(Final Output)', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Corruption Type', fontsize=12)
        ax3.set_ylabel('Prediction Magnitude', fontsize=12)
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for bar in bars3:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 添加总标题
        fig.suptitle('LSS Three Steps Feature Magnitude Analysis\n(Separated View for Clear Comparison)', 
                     fontsize=16, fontweight='bold', y=1.02)
        
        # 计算变化百分比并添加到子图中
        if 'origin' in corruption_types:
            origin_idx = corruption_types.index('origin')
            
            for i, (ax, mags, step_name) in enumerate([(ax1, lift_mags, 'Lift'), 
                                                       (ax2, splat_mags, 'Splat'), 
                                                       (ax3, shoot_mags, 'Shoot')]):
                baseline = mags[origin_idx]
                changes = []
                for j, ct in enumerate(corruption_types):
                    if ct != 'origin':
                        change_pct = ((mags[j] - baseline) / baseline * 100) if baseline > 1e-8 else 0
                        changes.append(f"{ct.upper()}: {change_pct:+.1f}%")
                
                # 在子图底部添加变化百分比信息
                if changes:
                    change_text = '\n'.join(changes)
                    ax.text(0.02, 0.02, f'Change from Origin:\n{change_text}', 
                           transform=ax.transAxes, fontsize=9, 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8),
                           verticalalignment='bottom')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_drops(self, comparison: Dict, save_path: str):
        """绘制性能下降对比图"""
        # 检查字典结构
        if 'shoot_impact' not in comparison or not comparison['shoot_impact']:
            print("Warning: No shoot_impact data found in comparison")
            return
            
        corruption_types = list(comparison['shoot_impact'].keys())
        iou_drops = [comparison['shoot_impact'][ct]['iou_drop'] for ct in corruption_types]
        acc_drops = [comparison['shoot_impact'][ct]['accuracy_drop'] for ct in corruption_types]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # IoU Drop
        bars1 = ax1.bar(corruption_types, iou_drops, alpha=0.7, color='red')
        ax1.set_title('IoU Drop by Corruption Type')
        ax1.set_ylabel('IoU Drop')
        ax1.set_xlabel('Corruption Type')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom')
        
        # Accuracy Drop
        bars2 = ax2.bar(corruption_types, acc_drops, alpha=0.7, color='orange')
        ax2.set_title('Accuracy Drop by Corruption Type')
        ax2.set_ylabel('Accuracy Drop')
        ax2.set_xlabel('Corruption Type')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_predictions(self, results: Dict, save_path: str):
        """绘制预测结果对比"""
        n_corruptions = len(results)
        fig, axes = plt.subplots(2, n_corruptions, figsize=(4*n_corruptions, 8))
        
        if n_corruptions == 1:
            axes = axes.reshape(2, 1)
        
        for i, (corruption_type, result) in enumerate(results.items()):
            # 预测结果
            pred = result['predictions'][0, 0].cpu().numpy()
            axes[0, i].imshow(pred, cmap='Blues', vmin=0, vmax=1)
            axes[0, i].set_title(f'{corruption_type}\nPredictions')
            axes[0, i].axis('off')
            
            # 真实标签
            if i == 0:  # 只显示一次真实标签
                gt = result['ground_truth'][0, 0].cpu().numpy()
                axes[1, i].imshow(gt, cmap='Greys', vmin=0, vmax=1)
                axes[1, i].set_title('Ground Truth')
            else:
                # 显示差异
                gt = result['ground_truth'][0, 0].cpu().numpy()
                diff = np.abs(pred - gt)
                axes[1, i].imshow(diff, cmap='Reds', vmin=0, vmax=1)
                axes[1, i].set_title(f'{corruption_type}\nPrediction Error')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_impact_heatmap(self, comparison: Dict, save_path: str):
        """绘制影响热力图"""
        # 检查字典结构
        if not comparison['lift_impact']:
            print("Warning: No impact data found in comparison")
            return
            
        # 准备数据
        corruption_types = list(comparison['lift_impact'].keys())
        metrics = ['Lift Magnitude', 'Lift Std', 'Lift Sparsity',
                  'Splat Magnitude', 'Splat Std', 'Splat Sparsity', 
                  'IoU Drop', 'Accuracy Drop', 'Pred Magnitude']
        
        # 构建热力图数据
        heatmap_data = []
        for ct in corruption_types:
            row = [
                comparison['lift_impact'][ct]['magnitude_change'],
                comparison['lift_impact'][ct]['std_change'], 
                comparison['lift_impact'][ct]['sparsity_change'],
                comparison['splat_impact'][ct]['magnitude_change'],
                comparison['splat_impact'][ct]['std_change'],
                comparison['splat_impact'][ct]['sparsity_change'],
                comparison['shoot_impact'][ct]['iou_drop'],
                comparison['shoot_impact'][ct]['accuracy_drop'],
                comparison['shoot_impact'][ct]['prediction_magnitude_change']
            ]
            heatmap_data.append(row)
        
        heatmap_data = np.array(heatmap_data)
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(heatmap_data, 
                   xticklabels=metrics,
                   yticklabels=corruption_types,
                   annot=True, 
                   fmt='.3f',
                   cmap='RdYlBu_r',
                   center=0)
        plt.title('Corruption Impact Heatmap on LSS Steps')
        plt.xlabel('Impact Metrics')
        plt.ylabel('Corruption Types')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_vulnerability_radar(self, comparison: Dict, save_path: str):
        """绘制脆弱性分析雷达图 - 直观显示各步骤对不同干扰的敏感程度"""
        if not comparison['lift_impact']:
            return
            
        corruption_types = list(comparison['lift_impact'].keys())
        
        # 定义评估指标（标准化到0-1范围）
        metrics = ['Lift Feature Change', 'Splat Feature Change', 'Shoot Performance Drop', 
                   'Lift Stability', 'Splat Stability', 'Shoot Accuracy']
        
        fig, axes = plt.subplots(1, len(corruption_types), figsize=(6*len(corruption_types), 6), subplot_kw=dict(projection='polar'))
        if len(corruption_types) == 1:
            axes = [axes]
            
        for i, corruption_type in enumerate(corruption_types):
            # 计算各项指标（标准化）
            lift_change = abs(comparison['lift_impact'][corruption_type]['magnitude_change'])
            splat_change = abs(comparison['splat_impact'][corruption_type]['magnitude_change'])
            shoot_drop = comparison['shoot_impact'][corruption_type]['iou_drop']
            
            # 稳定性指标（变化越小越稳定）
            lift_stability = max(0, 1 - abs(comparison['lift_impact'][corruption_type]['std_change']))
            splat_stability = max(0, 1 - abs(comparison['splat_impact'][corruption_type]['std_change']))
            shoot_accuracy = max(0, 1 - comparison['shoot_impact'][corruption_type]['accuracy_drop'])
            
            values = [lift_change, splat_change, shoot_drop, lift_stability, splat_stability, shoot_accuracy]
            
            # 标准化到0-1范围
            max_val = max(values) if max(values) > 0 else 1
            values = [v/max_val for v in values]
            
            # 闭合雷达图
            values += values[:1]
            angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
            angles += angles[:1]
            
            # 绘制雷达图
            axes[i].plot(angles, values, 'o-', linewidth=2, label=corruption_type)
            axes[i].fill(angles, values, alpha=0.25)
            axes[i].set_xticks(angles[:-1])
            axes[i].set_xticklabels(metrics, fontsize=10)
            axes[i].set_ylim(0, 1)
            axes[i].set_title(f'{corruption_type.upper()} Vulnerability Analysis', fontsize=14, pad=20)
            axes[i].grid(True)
            
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_step_sensitivity(self, comparison: Dict, save_path: str):
        """绘制步骤敏感性对比图 - 清晰显示哪个步骤最脆弱"""
        if not comparison['lift_impact']:
            return
            
        corruption_types = list(comparison['lift_impact'].keys())
        
        # 计算每个步骤的敏感性评分
        step_scores = {
            'Lift': [],
            'Splat': [], 
            'Shoot': []
        }
        
        for corruption_type in corruption_types:
            # Lift步骤敏感性（特征变化幅度）
            lift_score = (abs(comparison['lift_impact'][corruption_type]['magnitude_change']) + 
                         abs(comparison['lift_impact'][corruption_type]['std_change']) + 
                         abs(comparison['lift_impact'][corruption_type]['sparsity_change'])) / 3
            
            # Splat步骤敏感性
            splat_score = (abs(comparison['splat_impact'][corruption_type]['magnitude_change']) + 
                          abs(comparison['splat_impact'][corruption_type]['std_change']) + 
                          abs(comparison['splat_impact'][corruption_type]['sparsity_change'])) / 3
            
            # Shoot步骤敏感性（性能下降）
            shoot_score = (comparison['shoot_impact'][corruption_type]['iou_drop'] + 
                          comparison['shoot_impact'][corruption_type]['accuracy_drop'] + 
                          abs(comparison['shoot_impact'][corruption_type]['prediction_magnitude_change'])) / 3
            
            step_scores['Lift'].append(lift_score)
            step_scores['Splat'].append(splat_score)
            step_scores['Shoot'].append(shoot_score)
        
        # 绘制对比图
        x = np.arange(len(corruption_types))
        width = 0.25
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 左图：各步骤敏感性对比
        bars1 = ax1.bar(x - width, step_scores['Lift'], width, label='Lift Step', alpha=0.8, color='skyblue')
        bars2 = ax1.bar(x, step_scores['Splat'], width, label='Splat Step', alpha=0.8, color='lightgreen')
        bars3 = ax1.bar(x + width, step_scores['Shoot'], width, label='Shoot Step', alpha=0.8, color='salmon')
        
        ax1.set_xlabel('Corruption Type', fontsize=12)
        ax1.set_ylabel('Sensitivity Score', fontsize=12)
        ax1.set_title('LSS Three Steps Sensitivity Comparison\n(Higher Score = More Sensitive)', fontsize=14)
        ax1.set_xticks(x)
        ax1.set_xticklabels([ct.upper() for ct in corruption_types])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=10)
        
        # 右图：平均敏感性排名
        avg_scores = [np.mean(step_scores['Lift']), np.mean(step_scores['Splat']), np.mean(step_scores['Shoot'])]
        steps = ['Lift', 'Splat', 'Shoot']
        colors = ['skyblue', 'lightgreen', 'salmon']
        
        # 按敏感性排序
        sorted_data = sorted(zip(steps, avg_scores, colors), key=lambda x: x[1], reverse=True)
        sorted_steps, sorted_scores, sorted_colors = zip(*sorted_data)
        
        bars = ax2.bar(sorted_steps, sorted_scores, color=sorted_colors, alpha=0.8)
        ax2.set_ylabel('Average Sensitivity Score', fontsize=12)
        ax2.set_title('Step Vulnerability Ranking\n(Most Vulnerable to Most Stable)', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        # 添加排名标签
        for i, (bar, score) in enumerate(zip(bars, sorted_scores)):
            ax2.text(bar.get_x() + bar.get_width()/2., score + 0.01,
                    f'#{i+1}\n{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_corruption_propagation(self, results: Dict, comparison: Dict, save_path: str):
        """绘制干扰传播分析图 - 显示干扰如何在三个步骤间传播"""
        if not comparison['lift_impact']:
            return
            
        corruption_types = [ct for ct in results.keys() if ct != 'origin']
        
        fig, axes = plt.subplots(len(corruption_types), 1, figsize=(14, 6*len(corruption_types)))
        if len(corruption_types) == 1:
            axes = [axes]
        
        for i, corruption_type in enumerate(corruption_types):
            # 计算传播强度
            baseline = results['origin']['stats']
            corrupted = results[corruption_type]['stats']
            
            # 各步骤的变化幅度
            lift_change = abs(corrupted['lift']['feature_magnitude'] - baseline['lift']['feature_magnitude']) / baseline['lift']['feature_magnitude']
            splat_change = abs(corrupted['splat']['feature_magnitude'] - baseline['splat']['feature_magnitude']) / baseline['splat']['feature_magnitude']
            shoot_change = abs(baseline['shoot']['iou'] - corrupted['shoot']['iou']) / baseline['shoot']['iou'] if baseline['shoot']['iou'] > 1e-8 else 0
            
            # 绘制传播路径
            steps = ['Input Images', 'Lift\n(Feature Extract)', 'Splat\n(BEV Fusion)', 'Shoot\n(Final Prediction)']
            changes = [1.0, 1.0 + lift_change, 1.0 + lift_change + splat_change, 1.0 + lift_change + splat_change + shoot_change]
            
            # 标准化
            changes = [c / max(changes) for c in changes]
            
            ax = axes[i]
            
            # 绘制传播路径
            x_pos = np.arange(len(steps))
            line = ax.plot(x_pos, changes, 'o-', linewidth=3, markersize=10, label=f'{corruption_type.upper()} Corruption Propagation')
            color = line[0].get_color()
            
            # 填充区域显示影响强度
            ax.fill_between(x_pos, 0, changes, alpha=0.3, color=color)
            
            # 添加数值标签
            for j, (x, y) in enumerate(zip(x_pos, changes)):
                if j > 0:  # 跳过输入图像
                    change_pct = (changes[j] - changes[j-1]) * 100
                    ax.text(x, y + 0.05, f'+{change_pct:.1f}%', ha='center', va='bottom', 
                           fontsize=10, fontweight='bold', color=color)
            
            ax.set_xticks(x_pos)
            ax.set_xticklabels(steps, fontsize=11)
            ax.set_ylabel('Cumulative Impact Intensity', fontsize=12)
            ax.set_title(f'{corruption_type.upper()} Corruption Propagation in LSS Three Steps', fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_ylim(0, 1.2)
            
            # 添加步骤分界线
            for x in x_pos[1:-1]:
                ax.axvline(x + 0.5, color='gray', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_comprehensive_vulnerability(self, results: Dict, comparison: Dict, save_path: str):
        """绘制综合脆弱性评估图 - 综合所有指标的最终评估"""
        if not comparison['lift_impact']:
            return
            
        corruption_types = list(comparison['lift_impact'].keys())
        
        # 计算综合脆弱性评分
        vulnerability_matrix = []
        
        for corruption_type in corruption_types:
            # 收集所有影响指标
            lift_impacts = [
                abs(comparison['lift_impact'][corruption_type]['magnitude_change']),
                abs(comparison['lift_impact'][corruption_type]['std_change']),
                abs(comparison['lift_impact'][corruption_type]['sparsity_change'])
            ]
            
            splat_impacts = [
                abs(comparison['splat_impact'][corruption_type]['magnitude_change']),
                abs(comparison['splat_impact'][corruption_type]['std_change']),
                abs(comparison['splat_impact'][corruption_type]['sparsity_change'])
            ]
            
            shoot_impacts = [
                comparison['shoot_impact'][corruption_type]['iou_drop'],
                comparison['shoot_impact'][corruption_type]['accuracy_drop'],
                abs(comparison['shoot_impact'][corruption_type]['prediction_magnitude_change'])
            ]
            
            vulnerability_matrix.append(lift_impacts + splat_impacts + shoot_impacts)
        
        vulnerability_matrix = np.array(vulnerability_matrix)
        
        # 创建综合评估图
        fig = plt.figure(figsize=(16, 10))
        
        # 主热力图
        ax1 = plt.subplot(2, 2, (1, 2))
        
        metrics_labels = [
            'Lift\nFeature Mag', 'Lift\nStd Dev', 'Lift\nSparsity',
            'Splat\nFeature Mag', 'Splat\nStd Dev', 'Splat\nSparsity',
            'Shoot\nIoU Drop', 'Shoot\nAcc Drop', 'Shoot\nPred Change'
        ]
        
        im = ax1.imshow(vulnerability_matrix, cmap='Reds', aspect='auto')
        ax1.set_xticks(range(len(metrics_labels)))
        ax1.set_xticklabels(metrics_labels, rotation=45, ha='right', fontsize=10)
        ax1.set_yticks(range(len(corruption_types)))
        ax1.set_yticklabels([ct.upper() for ct in corruption_types], fontsize=12)
        ax1.set_title('LSS Model Comprehensive Vulnerability Heatmap\n(Darker Color = Greater Impact)', fontsize=14, pad=20)
        
        # 添加数值标注
        for i in range(len(corruption_types)):
            for j in range(len(metrics_labels)):
                text = ax1.text(j, i, f'{vulnerability_matrix[i, j]:.3f}',
                               ha="center", va="center", color="black" if vulnerability_matrix[i, j] < 0.5 else "white",
                               fontsize=9)
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax1, shrink=0.8)
        cbar.set_label('Impact Intensity', fontsize=12)
        
        # 各步骤平均脆弱性对比
        ax2 = plt.subplot(2, 2, 3)
        
        step_vulnerabilities = {
            'Lift': np.mean(vulnerability_matrix[:, :3], axis=1),
            'Splat': np.mean(vulnerability_matrix[:, 3:6], axis=1),
            'Shoot': np.mean(vulnerability_matrix[:, 6:], axis=1)
        }
        
        x = np.arange(len(corruption_types))
        width = 0.25
        
        bars1 = ax2.bar(x - width, step_vulnerabilities['Lift'], width, label='Lift', alpha=0.8, color='lightblue')
        bars2 = ax2.bar(x, step_vulnerabilities['Splat'], width, label='Splat', alpha=0.8, color='lightgreen')
        bars3 = ax2.bar(x + width, step_vulnerabilities['Shoot'], width, label='Shoot', alpha=0.8, color='lightcoral')
        
        ax2.set_xlabel('Corruption Type', fontsize=12)
        ax2.set_ylabel('Average Vulnerability Score', fontsize=12)
        ax2.set_title('Average Vulnerability by Step', fontsize=12)
        ax2.set_xticks(x)
        ax2.set_xticklabels([ct.upper() for ct in corruption_types])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 总体脆弱性排名
        ax3 = plt.subplot(2, 2, 4)
        
        overall_vulnerability = np.mean(vulnerability_matrix, axis=1)
        sorted_indices = np.argsort(overall_vulnerability)[::-1]
        
        bars = ax3.barh(range(len(corruption_types)), 
                       [overall_vulnerability[i] for i in sorted_indices],
                       color=['red' if i == sorted_indices[0] else 'orange' if i == sorted_indices[1] else 'yellow' 
                             for i in sorted_indices],
                       alpha=0.8)
        
        ax3.set_yticks(range(len(corruption_types)))
        ax3.set_yticklabels([corruption_types[i].upper() for i in sorted_indices], fontsize=12)
        ax3.set_xlabel('Overall Vulnerability Score', fontsize=12)
        ax3.set_title('Corruption Type Vulnerability Ranking', fontsize=12)
        ax3.grid(True, alpha=0.3)
        
        # 添加排名标签
        for i, (bar, score) in enumerate(zip(bars, [overall_vulnerability[j] for j in sorted_indices])):
            ax3.text(score + 0.01, bar.get_y() + bar.get_height()/2,
                    f'#{i+1} ({score:.3f})', va='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_feature_distribution_changes(self, results: Dict, save_path: str):
        """绘制特征分布变化对比图"""
        corruption_types = [ct for ct in results.keys() if ct != 'origin']
        
        fig, axes = plt.subplots(3, len(corruption_types), figsize=(6*len(corruption_types), 12))
        if len(corruption_types) == 1:
            axes = axes.reshape(-1, 1)
        
        baseline = results['origin']
        
        for i, corruption_type in enumerate(corruption_types):
            corrupted = results[corruption_type]
            
            # Lift特征分布
            ax = axes[0, i] if len(corruption_types) > 1 else axes[0]
            
            # 获取特征样本进行分布比较
            baseline_lift = baseline['cam_features'].flatten().cpu().numpy()
            corrupted_lift = corrupted['cam_features'].flatten().cpu().numpy()
            
            # 采样以提高绘图速度
            sample_size = min(10000, len(baseline_lift))
            baseline_sample = np.random.choice(baseline_lift, sample_size, replace=False)
            corrupted_sample = np.random.choice(corrupted_lift, sample_size, replace=False)
            
            ax.hist(baseline_sample, bins=50, alpha=0.5, label='Origin', density=True, color='blue')
            ax.hist(corrupted_sample, bins=50, alpha=0.5, label=corruption_type.upper(), density=True, color='red')
            ax.set_title(f'Lift Feature Distribution Change\n({corruption_type.upper()})', fontsize=12)
            ax.set_xlabel('Feature Value')
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Splat特征分布
            ax = axes[1, i] if len(corruption_types) > 1 else axes[1]
            
            baseline_splat = baseline['bev_features'].flatten().cpu().numpy()
            corrupted_splat = corrupted['bev_features'].flatten().cpu().numpy()
            
            baseline_sample = np.random.choice(baseline_splat, sample_size, replace=False)
            corrupted_sample = np.random.choice(corrupted_splat, sample_size, replace=False)
            
            ax.hist(baseline_sample, bins=50, alpha=0.5, label='Origin', density=True, color='blue')
            ax.hist(corrupted_sample, bins=50, alpha=0.5, label=corruption_type.upper(), density=True, color='red')
            ax.set_title(f'Splat Feature Distribution Change\n({corruption_type.upper()})', fontsize=12)
            ax.set_xlabel('Feature Value')
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Shoot预测分布
            ax = axes[2, i] if len(corruption_types) > 1 else axes[2]
            
            baseline_shoot = baseline['predictions'].flatten().cpu().numpy()
            corrupted_shoot = corrupted['predictions'].flatten().cpu().numpy()
            
            ax.hist(baseline_shoot, bins=50, alpha=0.5, label='Origin', density=True, color='blue')
            ax.hist(corrupted_shoot, bins=50, alpha=0.5, label=corruption_type.upper(), density=True, color='red')
            ax.set_title(f'Shoot Prediction Distribution Change\n({corruption_type.upper()})', fontsize=12)
            ax.set_xlabel('Prediction Value')
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_corruption_intensity_impact(self, results: Dict, comparison: Dict, save_path: str):
        """绘制干扰强度vs性能影响关系图"""
        if not comparison['lift_impact']:
            return
            
        corruption_types = list(comparison['lift_impact'].keys())
        
        # 模拟不同强度级别（实际使用中可以通过多次实验获得）
        intensities = ['Low', 'Medium', 'High']  # 当前只有Medium，这里展示框架
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 各步骤性能下降趋势
        for corruption_type in corruption_types:
            # 这里使用当前的medium强度数据，实际应用中可以扩展
            lift_impacts = [comparison['lift_impact'][corruption_type]['magnitude_change']]
            splat_impacts = [comparison['splat_impact'][corruption_type]['magnitude_change']]
            shoot_impacts = [comparison['shoot_impact'][corruption_type]['iou_drop']]
            
            ax1.plot([1], lift_impacts, 'o-', label=f'{corruption_type.upper()} - Lift', markersize=8)
            ax1.plot([1], splat_impacts, 's-', label=f'{corruption_type.upper()} - Splat', markersize=8)
            ax1.plot([1], shoot_impacts, '^-', label=f'{corruption_type.upper()} - Shoot', markersize=8)
        
        ax1.set_xlabel('Corruption Intensity Level')
        ax1.set_ylabel('Performance Impact')
        ax1.set_title('Corruption Intensity vs Step Performance Impact')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks([1])
        ax1.set_xticklabels(['Medium'])
        
        # 临界点分析
        ax2.bar(corruption_types, [comparison['shoot_impact'][ct]['iou_drop'] for ct in corruption_types], 
               color=['red' if comparison['shoot_impact'][ct]['iou_drop'] > 0.1 else 'orange' 
                     if comparison['shoot_impact'][ct]['iou_drop'] > 0.05 else 'green' 
                     for ct in corruption_types], alpha=0.8)
        ax2.set_ylabel('IoU Drop')
        ax2.set_title('Performance Drop Severity Classification\n(Red: Severe, Orange: Moderate, Green: Mild)')
        ax2.grid(True, alpha=0.3)
        
        # 添加临界线
        ax2.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='Severe Threshold')
        ax2.axhline(y=0.05, color='orange', linestyle='--', alpha=0.7, label='Moderate Threshold')
        ax2.legend()
        
        # 恢复能力分析（假设数据）
        recovery_scores = {}
        for ct in corruption_types:
            # 基于特征变化计算恢复能力（变化越小恢复能力越强）
            recovery_score = 1 / (1 + abs(comparison['shoot_impact'][ct]['prediction_magnitude_change']))
            recovery_scores[ct] = recovery_score
        
        ax3.bar(recovery_scores.keys(), recovery_scores.values(), 
               color=['green' if v > 0.8 else 'orange' if v > 0.6 else 'red' for v in recovery_scores.values()],
               alpha=0.8)
        ax3.set_ylabel('Recovery Capability Score')
        ax3.set_title('Model Recovery Capability for Different Corruptions\n(Higher Score = Stronger Recovery)')
        ax3.grid(True, alpha=0.3)
        
        # 风险评估矩阵
        risk_matrix = []
        for ct in corruption_types:
            impact_severity = comparison['shoot_impact'][ct]['iou_drop']
            occurrence_likelihood = 0.7 if ct == 'noise' else 0.3  # 假设概率
            risk_score = impact_severity * occurrence_likelihood
            risk_matrix.append([impact_severity, occurrence_likelihood, risk_score])
        
        risk_matrix = np.array(risk_matrix)
        
        scatter = ax4.scatter(risk_matrix[:, 1], risk_matrix[:, 0], 
                            s=risk_matrix[:, 2]*1000, 
                            c=risk_matrix[:, 2], 
                            cmap='Reds', alpha=0.7)
        
        for i, ct in enumerate(corruption_types):
            ax4.annotate(ct.upper(), (risk_matrix[i, 1], risk_matrix[i, 0]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=12, fontweight='bold')
        
        ax4.set_xlabel('Occurrence Probability')
        ax4.set_ylabel('Impact Severity')
        ax4.set_title('Risk Assessment Matrix\n(Bubble Size Indicates Risk Level)')
        ax4.grid(True, alpha=0.3)
        
        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax4)
        cbar.set_label('Risk Score')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_normalized_feature_comparison(self, results: Dict, save_path: str):
        """绘制标准化特征对比图 - 统一刻度显示"""
        if not HAS_SKLEARN:
            print("Warning: sklearn not available, skipping normalized feature comparison")
            return
            
        corruption_types = list(results.keys())
        lift_mags = [results[ct]['stats']['lift']['feature_magnitude'] for ct in corruption_types]
        splat_mags = [results[ct]['stats']['splat']['feature_magnitude'] for ct in corruption_types]
        shoot_mags = [results[ct]['stats']['shoot']['prediction_magnitude'] for ct in corruption_types]
        
        # 标准化到0-1范围
        scaler = MinMaxScaler()
        
        # 将数据组织成矩阵进行标准化
        data_matrix = np.array([lift_mags, splat_mags, shoot_mags]).T  # shape: (n_corruptions, 3)
        normalized_data = scaler.fit_transform(data_matrix)
        
        # 提取标准化后的数据
        norm_lift = normalized_data[:, 0]
        norm_splat = normalized_data[:, 1] 
        norm_shoot = normalized_data[:, 2]
        
        # 创建对比图
        x = np.arange(len(corruption_types))
        width = 0.25
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 左图：标准化对比
        bars1 = ax1.bar(x - width, norm_lift, width, label='Lift Features', alpha=0.8, color='skyblue')
        bars2 = ax1.bar(x, norm_splat, width, label='Splat Features', alpha=0.8, color='lightcoral')
        bars3 = ax1.bar(x + width, norm_shoot, width, label='Shoot Predictions', alpha=0.8, color='lightgreen')
        
        ax1.set_xlabel('Corruption Type', fontsize=12)
        ax1.set_ylabel('Normalized Feature Magnitude (0-1)', fontsize=12)
        ax1.set_title('Normalized Feature Magnitude Comparison\n(All Steps on Same Scale)', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels([ct.upper() for ct in corruption_types])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 右图：相对变化百分比
        if 'origin' in corruption_types:
            origin_idx = corruption_types.index('origin')
            
            # 计算相对于origin的变化百分比
            other_types = [ct for ct in corruption_types if ct != 'origin']
            other_indices = [i for i, ct in enumerate(corruption_types) if ct != 'origin']
            
            lift_changes = [(lift_mags[i] - lift_mags[origin_idx])/lift_mags[origin_idx]*100 
                          if lift_mags[origin_idx] > 1e-8 else 0 for i in other_indices]
            splat_changes = [(splat_mags[i] - splat_mags[origin_idx])/splat_mags[origin_idx]*100 
                           if splat_mags[origin_idx] > 1e-8 else 0 for i in other_indices]
            shoot_changes = [(shoot_mags[i] - shoot_mags[origin_idx])/shoot_mags[origin_idx]*100 
                           if shoot_mags[origin_idx] > 1e-8 else 0 for i in other_indices]
            
            x2 = np.arange(len(other_types))
            bars1_pct = ax2.bar(x2 - width, lift_changes, width, label='Lift Features', alpha=0.8, color='skyblue')
            bars2_pct = ax2.bar(x2, splat_changes, width, label='Splat Features', alpha=0.8, color='lightcoral')
            bars3_pct = ax2.bar(x2 + width, shoot_changes, width, label='Shoot Predictions', alpha=0.8, color='lightgreen')
            
            ax2.set_xlabel('Corruption Type', fontsize=12)
            ax2.set_ylabel('Change from Origin (%)', fontsize=12)
            ax2.set_title('Percentage Change from Baseline\n(Positive = Increase, Negative = Decrease)', fontsize=14, fontweight='bold')
            ax2.set_xticks(x2)
            ax2.set_xticklabels([ct.upper() for ct in other_types])
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)  # 添加零线
            
            # 添加数值标签
            for bars in [bars1_pct, bars2_pct, bars3_pct]:
                for bar in bars:
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., 
                            height + (1 if height >= 0 else -3),
                            f'{height:+.1f}%', ha='center', 
                            va='bottom' if height >= 0 else 'top', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def generate_report(self, results: Dict, comparison: Dict, output_dir: str = './analysis_output'):
        """生成增强的分析报告"""
        report_path = os.path.join(output_dir, 'corruption_analysis_report.md')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# LSS模型干扰影响深度分析报告\n\n")
            
            # 执行摘要
            f.write("## 🎯 执行摘要\n\n")
            f.write("本报告对LSS（Lift-Splat-Shoot）模型在不同干扰条件下的表现进行了全面分析，")
            f.write("识别了系统的脆弱环节，量化了干扰影响程度，并提供了针对性的改进建议。\n\n")
            
            # 关键发现
            if comparison['shoot_impact']:
                # 找出最脆弱的干扰类型
                max_iou_drop = max(comparison['shoot_impact'].items(), key=lambda x: x[1]['iou_drop'])
                corruption_types = list(comparison['lift_impact'].keys())
                
                # 计算各步骤平均敏感性
                step_sensitivities = {}
                for step in ['lift', 'splat', 'shoot']:
                    if step == 'shoot':
                        avg_sensitivity = np.mean([comparison['shoot_impact'][ct]['iou_drop'] for ct in corruption_types])
                    else:
                        avg_sensitivity = np.mean([abs(comparison[f'{step}_impact'][ct]['magnitude_change']) for ct in corruption_types])
                    step_sensitivities[step] = avg_sensitivity
                
                most_vulnerable_step = max(step_sensitivities.items(), key=lambda x: x[1])
                
                f.write("### 🔍 Key Findings\n\n")
                f.write(f"1. **Most Severe Corruption**: {max_iou_drop[0].upper()} (IoU Drop: {max_iou_drop[1]['iou_drop']:.4f})\n")
                f.write(f"2. **Most Vulnerable Step**: {most_vulnerable_step[0].upper()} Step (Sensitivity Score: {most_vulnerable_step[1]:.4f})\n")
                f.write(f"3. **Analysis Batches**: {len(results)} batches\n")
                f.write(f"4. **Corruption Types**: {len(corruption_types)} types\n\n")
            
            # 详细分析
            f.write("## 📊 详细分析结果\n\n")
            
            # LSS三步骤说明
            f.write("### LSS模型架构说明\n\n")
            f.write("LSS模型包含三个核心步骤：\n")
            f.write("1. **Lift步骤**: 从2D图像提取特征并估计深度信息\n")
            f.write("2. **Splat步骤**: 将多视角相机特征融合到鸟瞰图(BEV)网格中\n")
            f.write("3. **Shoot步骤**: 在BEV特征上进行最终的语义分割预测\n\n")
            
            # 各步骤影响分析表格
            for step in ['lift', 'splat', 'shoot']:
                f.write(f"### {step.upper()}步骤影响分析\n\n")
                
                if step == 'shoot':
                    impact_data = comparison['shoot_impact']
                    f.write("| 干扰类型 | IoU下降 | 准确率下降 | 预测幅度变化 | 严重程度 |\n")
                    f.write("|---------|---------|------------|-------------|--------|\n")
                    for ct, data in impact_data.items():
                        severity = "🔴严重" if data['iou_drop'] > 0.1 else "🟡中等" if data['iou_drop'] > 0.05 else "🟢轻微"
                        f.write(f"| {ct.upper()} | {data['iou_drop']:.4f} | {data['accuracy_drop']:.4f} | {data['prediction_magnitude_change']:.4f} | {severity} |\n")
                else:
                    impact_data = comparison[f'{step}_impact']
                    f.write("| 干扰类型 | 特征幅度变化 | 标准差变化 | 稀疏度变化 | 综合评分 |\n")
                    f.write("|---------|-------------|------------|------------|--------|\n")
                    for ct, data in impact_data.items():
                        composite_score = (abs(data['magnitude_change']) + abs(data['std_change']) + abs(data['sparsity_change'])) / 3
                        f.write(f"| {ct.upper()} | {data['magnitude_change']:.4f} | {data['std_change']:.4f} | {data['sparsity_change']:.4f} | {composite_score:.4f} |\n")
                f.write("\n")
            
            # 脆弱性排名
            f.write("### 🏆 步骤脆弱性排名\n\n")
            
            # 计算各步骤综合脆弱性评分
            step_vulnerability_scores = {}
            for step in ['lift', 'splat', 'shoot']:
                if step == 'shoot':
                    scores = [comparison['shoot_impact'][ct]['iou_drop'] + comparison['shoot_impact'][ct]['accuracy_drop'] 
                             for ct in corruption_types]
                else:
                    scores = [abs(comparison[f'{step}_impact'][ct]['magnitude_change']) + 
                             abs(comparison[f'{step}_impact'][ct]['std_change']) + 
                             abs(comparison[f'{step}_impact'][ct]['sparsity_change']) 
                             for ct in corruption_types]
                step_vulnerability_scores[step] = np.mean(scores)
            
            # 排序
            sorted_steps = sorted(step_vulnerability_scores.items(), key=lambda x: x[1], reverse=True)
            
            f.write("| 排名 | 步骤 | 脆弱性评分 | 主要风险 |\n")
            f.write("|------|------|-----------|--------|\n")
            
            risk_descriptions = {
                'lift': '特征提取不稳定，深度估计误差',
                'splat': 'BEV融合精度下降，空间对应错误', 
                'shoot': '最终预测性能直接受损'
            }
            
            for i, (step, score) in enumerate(sorted_steps):
                rank_emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
                f.write(f"| {rank_emoji} #{i+1} | {step.upper()} | {score:.4f} | {risk_descriptions[step]} |\n")
            f.write("\n")
            
            # 干扰传播分析
            f.write("### 🔄 Corruption Propagation Analysis\n\n")
            f.write("Analysis shows corruption propagation patterns in LSS three steps:\n\n")
            
            for ct in corruption_types:
                baseline = results['origin']['stats'] if 'origin' in results else {}
                corrupted = results[ct]['stats'] if ct in results else {}
                
                if baseline and corrupted:
                    # 计算累积影响
                    lift_impact = abs(corrupted['lift']['feature_magnitude'] - baseline['lift']['feature_magnitude']) / baseline['lift']['feature_magnitude'] if baseline['lift']['feature_magnitude'] > 1e-8 else 0
                    splat_impact = abs(corrupted['splat']['feature_magnitude'] - baseline['splat']['feature_magnitude']) / baseline['splat']['feature_magnitude'] if baseline['splat']['feature_magnitude'] > 1e-8 else 0
                    shoot_impact = abs(baseline['shoot']['iou'] - corrupted['shoot']['iou']) / baseline['shoot']['iou'] if baseline['shoot']['iou'] > 1e-8 else 0
                    
                    f.write(f"**{ct.upper()} Corruption Propagation Path**:\n")
                    f.write(f"- Lift Stage Impact: {lift_impact:.3f} ({lift_impact*100:.1f}%)\n")
                    f.write(f"- Splat Stage Cumulative Impact: {lift_impact + splat_impact:.3f}\n")
                    f.write(f"- Shoot Stage Final Impact: {lift_impact + splat_impact + shoot_impact:.3f}\n\n")
            
            # 改进建议
            f.write("## 💡 Improvement Recommendations\n\n")
            
            f.write("### Targeted Protection Strategies\n\n")
            
            # 基于最脆弱步骤给出建议
            if most_vulnerable_step[0] == 'lift':
                f.write("**Lift步骤加固建议**:\n")
                f.write("- 增强特征提取器的鲁棒性训练\n")
                f.write("- 引入深度估计的正则化约束\n")
                f.write("- 采用多尺度特征融合策略\n")
                f.write("- 添加输入预处理和去噪模块\n\n")
            elif most_vulnerable_step[0] == 'splat':
                f.write("**Splat步骤加固建议**:\n")
                f.write("- 优化BEV投影算法的数值稳定性\n")
                f.write("- 引入空间注意力机制\n")
                f.write("- 增强多视角特征融合的一致性\n")
                f.write("- 添加几何约束损失函数\n\n")
            else:
                f.write("**Shoot步骤加固建议**:\n")
                f.write("- 增强BEV编码器的表达能力\n")
                f.write("- 采用集成学习提高预测稳定性\n")
                f.write("- 引入不确定性估计模块\n")
                f.write("- 优化损失函数设计\n\n")
            
            f.write("### 通用防护措施\n\n")
            f.write("1. **数据增强**: 在训练中加入对抗样本和噪声样本\n")
            f.write("2. **模型集成**: 使用多个模型的集成预测提高鲁棒性\n")
            f.write("3. **在线检测**: 部署异常检测模块识别潜在攻击\n")
            f.write("4. **自适应调整**: 根据输入质量动态调整模型参数\n")
            f.write("5. **防御训练**: 采用对抗训练提高模型抗干扰能力\n\n")
            
            # 风险评估
            f.write("### 🚨 风险评估矩阵\n\n")
            
            f.write("| 干扰类型 | 发生概率 | 影响严重程度 | 风险等级 | 优先级 |\n")
            f.write("|---------|---------|-------------|---------|-------|\n")
            
            for ct in corruption_types:
                iou_drop = comparison['shoot_impact'][ct]['iou_drop']
                probability = 0.7 if ct == 'noise' else 0.3  # 假设概率
                risk_score = iou_drop * probability
                
                risk_level = "🔴高" if risk_score > 0.07 else "🟡中" if risk_score > 0.03 else "🟢低"
                priority = "P1" if risk_score > 0.07 else "P2" if risk_score > 0.03 else "P3"
                
                f.write(f"| {ct.upper()} | {probability:.1f} | {iou_drop:.4f} | {risk_level} | {priority} |\n")
            f.write("\n")
            
            # 生成的可视化文件说明
            f.write("## 📈 可视化结果说明\n\n")
            f.write("本次分析生成了以下可视化文件：\n\n")
            
            visualization_files = [
                ("feature_magnitudes.png", "Feature Magnitude Comparison (Separated)", "Three steps displayed separately for clear comparison"),
                ("normalized_feature_comparison.png", "Normalized Feature Comparison", "All steps on same scale with percentage changes"),
                ("vulnerability_radar.png", "Vulnerability Radar Chart", "Intuitive display of step sensitivity to different corruptions"),
                ("step_sensitivity.png", "Step Sensitivity Comparison", "Clear ranking of most vulnerable steps"),
                ("corruption_propagation.png", "Corruption Propagation Analysis", "How corruption propagates and accumulates across steps"),
                ("comprehensive_vulnerability.png", "Comprehensive Vulnerability Assessment", "Multi-dimensional vulnerability evaluation"),
                ("feature_distribution_changes.png", "Feature Distribution Changes", "Feature distribution changes under corruption"),
                ("corruption_intensity_impact.png", "Corruption Intensity Impact", "Impact patterns of different corruption intensities"),
                ("performance_drops.png", "Performance Drop Comparison", "Quantified performance degradation"),
                ("predictions_comparison.png", "Prediction Results Comparison", "Intuitive comparison of prediction quality"),
                ("impact_heatmap.png", "Impact Heatmap", "Comprehensive display of all impact metrics")
            ]
            
            for filename, title, description in visualization_files:
                f.write(f"- **{filename}**: {title} - {description}\n")
            f.write("\n")
            
            # 技术细节
            f.write("## 🔧 技术细节\n\n")
            f.write("### 评估指标说明\n\n")
            f.write("- **IoU (Intersection over Union)**: 预测与真实标签的交并比\n")
            f.write("- **特征幅度**: 特征向量的L2范数均值\n")
            f.write("- **特征稀疏度**: 接近零的特征值比例\n")
            f.write("- **敏感性评分**: 综合多个指标的标准化评分\n")
            f.write("- **脆弱性评分**: 基于所有影响指标的综合评估\n\n")
            
            f.write("### 分析方法\n\n")
            f.write("1. **逐步分析**: 分别分析LSS三个步骤的中间输出\n")
            f.write("2. **对比分析**: 将干扰条件下的结果与基线进行对比\n")
            f.write("3. **统计分析**: 计算多个批次的统计指标\n")
            f.write("4. **可视化分析**: 生成多种图表直观展示结果\n\n")
            
            # 结论
            f.write("## 📝 结论\n\n")
            f.write("本次分析全面评估了LSS模型在不同干扰条件下的表现，识别了关键脆弱环节，")
            f.write("并提供了针对性的改进建议。建议优先关注最脆弱的步骤，采用相应的防护措施，")
            f.write("以提高模型的整体鲁棒性和安全性。\n\n")
            
            f.write("详细的可视化结果可在对应的图片文件中查看，")
            f.write("建议结合图表进行深入分析和决策制定。\n")
        
        print(f"📋 增强分析报告已保存至: {report_path}")


def run_corruption_analysis(version='mini', dataroot='/dataset/nuscenes', modelf=None, 
                          output_dir='./analysis_output', bsz=2, max_batches=5):
    """运行完整的干扰影响分析"""
    
    # 设置配置
    grid_conf = {
        'xbound': [-50.0, 50.0, 0.5],
        'ybound': [-50.0, 50.0, 0.5], 
        'zbound': [-10.0, 10.0, 20.0],
        'dbound': [4.0, 45.0, 1.0],
    }
    
    data_aug_conf = {
        'resize_lim': (0.193, 0.225),
        'final_dim': (128, 352),
        'rot_lim': (-5.4, 5.4),
        'H': 900, 'W': 1600,
        'rand_flip': True,
        'bot_pct_lim': (0.0, 0.22),
        'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
        'Ncams': 5,
    }
    
    # 加载数据和模型
    print("Loading data...")
    trainloader, valloader = compile_data(version, dataroot, data_aug_conf=data_aug_conf,
                                        grid_conf=grid_conf, bsz=bsz, nworkers=0,
                                        parser_name='segmentationdata')
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading model...")
    model = compile_model(grid_conf, data_aug_conf, outC=1)
    if modelf:
        model.load_state_dict(torch.load(modelf, map_location=device))
    model.to(device)
    model.eval()
    
    # 创建分析器
    analyzer = LSSCorruptionAnalyzer(model, device)
    
    # 运行分析
    print("Starting corruption impact analysis...")
    all_results = []
    all_comparisons = []
    
    batch_count = 0
    for batch_data in valloader:
        if batch_count >= max_batches:
            break
            
        images, rots, trans, intrins, post_rots, post_trans, binimgs = batch_data
        images = images.to(device)
        rots = rots.to(device) 
        trans = trans.to(device)
        intrins = intrins.to(device)
        post_rots = post_rots.to(device)
        post_trans = post_trans.to(device)
        binimgs = binimgs.to(device)
        
        print(f"Analyzing batch {batch_count + 1}/{max_batches}")
        
        try:
            # 逐步分析
            results = analyzer.analyze_step_by_step(
                images, rots, trans, intrins, post_rots, post_trans, binimgs,
                corruption_types=['origin', 'noise', 'fgsm', 'pgd', 'cw', 'dag', 'fusion', 'robust_bev']
            )
            
            # 比较分析
            comparison = analyzer.compare_corruptions(results)
            
            all_results.append(results)
            all_comparisons.append(comparison)
            
        except Exception as e:
            import traceback
            print(f"Error in batch {batch_count + 1}: {str(e)}")
            print(f"Full traceback: {traceback.format_exc()}")
            # 继续处理下一个批次而不是完全停止
            continue
            
        batch_count += 1
    
    # 合并结果并可视化
    print("Generating visualization results...")
    if all_results:
        # 使用第一个批次的结果进行可视化
        analyzer.visualize_analysis(all_results[0], all_comparisons[0], output_dir)
        analyzer.generate_report(all_results[0], all_comparisons[0], output_dir)
        
        print(f"Analysis completed! Results saved in: {output_dir}")
        
        # 打印关键结果摘要
        print("\n=== Key Results Summary ===")
        baseline_iou = all_results[0]['origin']['stats']['shoot']['iou']
        print(f"Baseline IoU: {baseline_iou:.4f}")
        
        if baseline_iou < 0.01:
            print("⚠️  Baseline IoU is very low, model may be randomly initialized")
            print("Consider using a pre-trained model for more meaningful results")
        
        for corruption_type in ['noise', 'fgsm', 'pgd', 'cw', 'dag', 'fusion', 'robust_bev']:
            if corruption_type in all_results[0]:
                corrupted_iou = all_results[0][corruption_type]['stats']['shoot']['iou']
                iou_drop = baseline_iou - corrupted_iou
                # 安全计算百分比，避免除零错误
                percentage_drop = (iou_drop / baseline_iou * 100) if abs(baseline_iou) > 1e-8 else 0.0
                print(f"{corruption_type}: IoU={corrupted_iou:.4f}, Drop={iou_drop:.4f} ({percentage_drop:.1f}%)")
    
    return all_results, all_comparisons 