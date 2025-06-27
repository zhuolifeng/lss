#!/usr/bin/env python3
"""
LSS模型三步骤干扰影响分析示例脚本
演示如何分析Lift、Splat、Shoot在不同干扰下的表现
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.analyzer_class import run_corruption_analysis


def main(dataroot='/dataset/nuscenes', modelf=None, output_dir='./lss_corruption_analysis', 
         bsz=1, max_batches=3):
    """运行LSS三步骤干扰影响分析"""
    
    print("=== LSS模型三步骤干扰影响分析 ===")
    print()
    print("分析目标:")
    print("1. Lift步骤: 图像特征提取和深度估计")
    print("2. Splat步骤: 多相机特征融合到BEV")
    print("3. Shoot步骤: BEV特征编码和最终预测")
    print()
    print("干扰类型:")
    print("- origin: 无干扰基线")
    print("- noise: 轻微噪声干扰")
    print("- fgsm: FGSM对抗攻击")
    print("- pgd: PGD对抗攻击") 
    print("- cw: C&W对抗攻击")
    print("- dag: DAG稠密对抗攻击")
    print()
    
    # 检查数据路径
    if not os.path.exists(dataroot):
        print(f"⚠️  数据路径不存在: {dataroot}")
        print("请使用正确的数据路径:")
        print("python analyze_lss_corruption.py --dataroot /your/path/to/nuscenes")
        return None, None
    
    try:
        # 运行分析
        results, comparisons = run_corruption_analysis(
            version='mini',
            dataroot=dataroot,
            modelf=modelf,
            output_dir=output_dir,
            bsz=bsz,
            max_batches=max_batches
        )
        
        if results:
            print("\n✅ 分析完成！")
            print(f"📊 结果保存在: {output_dir}")
            print("\n生成的文件包括:")
            print("- feature_magnitudes.png: 三步骤特征幅度对比")
            print("- performance_drops.png: 性能下降对比") 
            print("- predictions_comparison.png: 预测结果对比")
            print("- impact_heatmap.png: 影响热力图")
            print("- corruption_analysis_report.md: 详细分析报告")
            
            # 显示关键结果摘要
            baseline_iou = results[0]['origin']['stats']['shoot']['iou'] if results else 0.0
            print(f"\n=== 关键结果摘要 ===")
            print(f"基线IoU: {baseline_iou:.4f}")
            
            if baseline_iou < 0.01:
                print("\n⚠️  注意：检测到基线IoU非常低")
                print("这表明模型可能是随机初始化的，建议：")
                print("1. 使用预训练模型: --modelf /path/to/trained_model.pt")
                print("2. 或者先训练模型")
                print("3. 当前分析仍然有效，显示了网络架构对干扰的响应")
                
            for corruption_type in ['noise', 'fgsm', 'pgd', 'cw', 'dag']:
                if corruption_type in results[0]:
                    corrupted_iou = results[0][corruption_type]['stats']['shoot']['iou']
                    iou_drop = baseline_iou - corrupted_iou
                    percentage_drop = safe_percentage(iou_drop, baseline_iou)
                    print(f"{corruption_type}: IoU={corrupted_iou:.4f}, 下降={iou_drop:.4f} ({percentage_drop:.1f}%)")
            
        else:
            print("❌ 分析失败，未获得有效结果")
            
        return results, comparisons
        
    except Exception as e:
        import traceback
        print(f"❌ 分析过程出错: {str(e)}")
        print(f"详细错误信息: {traceback.format_exc()}")
        print("\n可能的解决方案:")
        print("1. 检查数据路径是否正确")
        print("2. 确保已安装所需依赖: matplotlib, seaborn, sklearn")
        print("3. 检查GPU内存是否足够")
        print("4. 考虑使用预训练模型以获得更有意义的结果")
        return None, None


def safe_percentage(numerator, denominator):
    """安全计算百分比，避免除零错误"""
    if abs(denominator) < 1e-8:
        return 0.0
    return (numerator / denominator) * 100


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='LSS三步骤干扰影响分析')
    parser.add_argument('--dataroot', default='/dataset/nuscenes', 
                       help='NuScenes数据集路径')
    parser.add_argument('--modelf', default='model525000.pt',
                       help='预训练模型路径 (可选)')
    parser.add_argument('--output_dir', default='./lss_corruption_analysis_all',
                       help='输出目录')
    parser.add_argument('--bsz', type=int, default=1,
                       help='批量大小')
    parser.add_argument('--max_batches', type=int, default=3,
                       help='分析的批次数')
    
    args = parser.parse_args()
    
    # 直接调用main函数，避免重复代码
    results, comparisons = main(
        dataroot=args.dataroot,
        modelf=args.modelf,
        output_dir=args.output_dir,
        bsz=args.bsz,
        max_batches=args.max_batches
    ) 