import torch
import torch.nn.functional as F
import random
from typing import Optional, Callable


class ClassicAdversarialAttack:
    """
    经典版本的对抗攻击实现 - 简化版，直接在原始图像空间操作
    """
    def __init__(self, model):
        self.model = model

    def classic_fgsm_attack(self, images, rots, trans, intrins, post_rots, post_trans, binimgs,
                           epsilon=2/255):
        """
        经典FGSM攻击实现 - 简化版，直接在原始图像空间操作
        """
        print(f"[经典FGSM] epsilon: {epsilon:.6f}")
        
        # 确保输入张量的维度正确
        # images: (S, C, H, W) -> (1, S, C, H, W) 添加批次维度
        if len(images.shape) == 4:
            images = images.unsqueeze(0)  # 添加批次维度
        if len(rots.shape) == 2:
            rots = rots.unsqueeze(0)
        if len(trans.shape) == 2:
            trans = trans.unsqueeze(0)
        if len(intrins.shape) == 3:
            intrins = intrins.unsqueeze(0)
        if len(post_rots.shape) == 3:
            post_rots = post_rots.unsqueeze(0)
        if len(post_trans.shape) == 2:
            post_trans = post_trans.unsqueeze(0)
        if len(binimgs.shape) == 3:
            binimgs = binimgs.unsqueeze(0)
        
        target = binimgs.float()
        
        # 克隆图像并设置梯度计算
        input_images = images.clone().detach().requires_grad_(True)
        
        # 前向传播
        outputs = self.model(input_images, rots, trans, intrins, post_rots, post_trans)
        
        # 计算损失
        pred = outputs.sigmoid()
        
        # 1. 二元交叉熵损失
        loss_ce = F.binary_cross_entropy_with_logits(outputs, target)
        
        # 2. IoU损失 (我们希望最小化IoU来降低分割准确性)
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection
        iou = intersection / (union + 1e-8)
        loss_iou = iou.mean()
        
        # 3. 车辆抑制损失 (鼓励在车辆区域产生错误预测)
        loss_suppress = (pred * target).mean()
        
        # 组合损失：BCE + 反向IoU + 车辆抑制
        total_loss = loss_ce - 2.0 * loss_iou + 1.5 * loss_suppress
        
        # 反向传播计算梯度
        total_loss.backward()
        
        # 生成对抗样本
        with torch.no_grad():
            sign_data_grad = input_images.grad.sign()
            perturbed_image = input_images + epsilon * sign_data_grad
            
            # 统计信息
            final_pred = self.model(perturbed_image, rots, trans, intrins, post_rots, post_trans).sigmoid()
            final_intersection = (final_pred * target).sum(dim=(2, 3))
            final_union = final_pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - final_intersection
            final_iou = (final_intersection / (final_union + 1e-8)).mean().item()
            orig_iou = loss_iou.item()
            print(f"[经典FGSM] 原始IoU: {orig_iou:.4f} -> 攻击后IoU: {final_iou:.4f}")
             
        # 返回时移除批次维度
        return perturbed_image.squeeze(0)

    def classic_pgd_attack(self, images, rots, trans, intrins, post_rots, post_trans, binimgs,
                          epsilon=2/255, alpha=0.5/255, num_steps=10):
        """
        经典PGD攻击实现 - 简化版，直接在原始图像空间操作
        """
        print(f"[经典PGD] epsilon: {epsilon:.6f}, alpha: {alpha:.6f}, steps: {num_steps}")
        
        # 确保输入张量的维度正确
        # images: (S, C, H, W) -> (1, S, C, H, W) 添加批次维度
        if len(images.shape) == 4:
            images = images.unsqueeze(0)  # 添加批次维度
        if len(rots.shape) == 2:
            rots = rots.unsqueeze(0)
        if len(trans.shape) == 2:
            trans = trans.unsqueeze(0)
        if len(intrins.shape) == 3:
            intrins = intrins.unsqueeze(0)
        if len(post_rots.shape) == 3:
            post_rots = post_rots.unsqueeze(0)
        if len(post_trans.shape) == 2:
            post_trans = post_trans.unsqueeze(0)
        if len(binimgs.shape) == 3:
            binimgs = binimgs.unsqueeze(0)
        
        target = binimgs.float()
        
        # 克隆原始图像并分离计算图，避免梯度冲突
        original_images = images.clone().detach()
        
        # 随机初始化扰动
        delta = torch.zeros_like(original_images).uniform_(-epsilon, epsilon) 
        delta.requires_grad_(True)
        
        # 计算原始IoU
        with torch.no_grad():
            original_output = self.model(original_images, rots, trans, intrins, post_rots, post_trans)
            orig_pred = original_output.sigmoid()
            orig_intersection = (orig_pred * target).sum(dim=(2, 3))
            orig_union = orig_pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - orig_intersection
            orig_iou = (orig_intersection / (orig_union + 1e-8)).mean().item()
            print(f"[经典PGD] 原始IoU: {orig_iou:.4f}")

        # PGD迭代
        for step in range(num_steps):
            # 清除之前的梯度
            if delta.grad is not None:
                delta.grad.zero_()
            
            # 创建当前的对抗样本（每次都重新创建计算图）
            perturbed_images = original_images + delta
            
            # 前向传播
            output = self.model(perturbed_images, rots, trans, intrins, post_rots, post_trans)
            
            # 使用IoU导向的损失
            pred = output.sigmoid()
            
            # 1. 二元交叉熵
            loss_ce = F.binary_cross_entropy_with_logits(output, target)
            
            # 2. IoU损失
            intersection = (pred * target).sum(dim=(2, 3))
            union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection
            iou = intersection / (union + 1e-8)
            loss_iou = iou.mean()
            
            # 3. 车辆抑制损失
            loss_suppress = (pred * target).mean()
            
            # 4. 非车辆区域假阳性损失
            non_vehicle_mask = 1.0 - target
            loss_false_positive = -(pred * non_vehicle_mask).mean()
            
            # 组合损失
            total_loss = loss_ce - 3.0 * loss_iou + 2.0 * loss_suppress + 1.0 * loss_false_positive
            
            # 反向传播（只针对delta）
            total_loss.backward()
            
            # PGD更新：delta = delta + alpha * sign(gradient)
            with torch.no_grad():
                delta.data = delta.data + alpha * delta.grad.sign()
                
                # 投影到L∞球内
                delta.data = torch.clamp(delta.data, -epsilon, epsilon)
            
            if step % 3 == 0:
                current_iou = iou.mean().item()
                print(f"[经典PGD] Step {step}: IoU = {current_iou:.4f}")
        
        # 生成最终对抗样本
        with torch.no_grad():
            final_perturbed_image = original_images + delta
            
            # 计算最终IoU
            final_output = self.model(final_perturbed_image, rots, trans, intrins, post_rots, post_trans)
            final_pred = final_output.sigmoid()
            final_intersection = (final_pred * target).sum(dim=(2, 3))
            final_union = final_pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - final_intersection
            final_iou = (final_intersection / (final_union + 1e-8)).mean().item()
            print(f"[经典PGD] 最终IoU: {final_iou:.4f} (下降: {orig_iou - final_iou:.4f})")
            
            # 扰动统计
            diff = delta.abs()
            max_diff = diff.max()
            mean_diff = diff.mean()
            
            print(f"[经典PGD] 扰动统计 - 最大: {max_diff:.6f}, 平均: {mean_diff:.6f}")
        
        # 返回时移除批次维度
        return final_perturbed_image.squeeze(0)

    def c_w_attack(self, images, rots, trans, intrins, post_rots, post_trans, binimgs,
                   c=1.0, kappa=0.2, max_iter=100, learning_rate=0.01, epsilon=8/255):
        """
        经典C&W攻击实现 - 简化版，避免tanh变换导致的图像变暗问题
        
        C&W攻击原理：
        - 将对抗样本生成转化为优化问题：minimize ||δ||_2 + c * f(x+δ)
        - 直接优化扰动δ，避免复杂的变量替换
        - f(x+δ) 是攻击目标函数，当攻击成功时 f(x+δ) ≤ 0
        - 通过epsilon约束确保扰动不可见
        """
        print(f"[经典C&W] c: {c}, kappa: {kappa}, max_iter: {max_iter}, epsilon: {epsilon:.6f}")
        
        # 确保输入张量的维度正确
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        if len(rots.shape) == 2:
            rots = rots.unsqueeze(0)
        if len(trans.shape) == 2:
            trans = trans.unsqueeze(0)
        if len(intrins.shape) == 3:
            intrins = intrins.unsqueeze(0)
        if len(post_rots.shape) == 3:
            post_rots = post_rots.unsqueeze(0)
        if len(post_trans.shape) == 2:
            post_trans = post_trans.unsqueeze(0)
        if len(binimgs.shape) == 3:
            binimgs = binimgs.unsqueeze(0)
        
        batch_size = images.size(0)
        target_probs = binimgs.float()
        
        # 保存原始图像，不改变其数值范围
        original_images = images.clone().detach()
        
        # 检测图像的实际数值范围
        img_min = original_images.min().item()
        img_max = original_images.max().item()
        print(f"[经典C&W] 图像数值范围: [{img_min:.3f}, {img_max:.3f}]")
        
        # 计算原始IoU作为基线
        with torch.no_grad():
            orig_outputs = self.model(original_images, rots, trans, intrins, post_rots, post_trans)
            orig_probs = torch.sigmoid(orig_outputs)
            orig_intersection = (orig_probs * target_probs).sum(dim=(2, 3))
            orig_union = orig_probs.sum(dim=(2, 3)) + target_probs.sum(dim=(2, 3)) - orig_intersection
            orig_iou = (orig_intersection / (orig_union + 1e-8)).mean().item()
            print(f"[经典C&W] 原始IoU: {orig_iou:.4f}")
        
        # 修复的C&W变量替换：直接优化扰动δ，然后投影到epsilon球内
        # 初始化小的随机扰动
        delta = torch.zeros_like(original_images, requires_grad=True)
        optimizer = torch.optim.Adam([delta], lr=learning_rate)
        
        best_l2 = float('inf')
        best_attack = original_images.clone()
        best_iou = orig_iou
        
        print(f"[经典C&W] 开始优化...")
        
        for iteration in range(max_iter):
            optimizer.zero_grad()
            
            # 生成对抗样本：原始图像 + 限制幅度的扰动
            clamped_delta = torch.clamp(delta, -epsilon, epsilon)
            adv_images = original_images + clamped_delta
            
            # 根据原始图像的数值范围进行合理的限制
            # 不强制限制到[0,1]，而是保持在合理范围内
            adv_images = torch.clamp(adv_images, img_min - epsilon, img_max + epsilon)
            
            # 前向传播
            outputs = self.model(adv_images, rots, trans, intrins, post_rots, post_trans)
            probs = torch.sigmoid(outputs)
            
            # 计算IoU
            intersection = (probs * target_probs).sum(dim=(2, 3))
            union = probs.sum(dim=(2, 3)) + target_probs.sum(dim=(2, 3)) - intersection
            iou = intersection / (union + 1e-8)
            mean_iou = iou.mean()
            
            # L2距离（实际扰动）
            actual_perturbation = adv_images - original_images
            l2_dist = torch.norm(actual_perturbation.view(batch_size, -1), dim=1)
            
            # C&W损失函数
            # f(x) = IoU - target_iou，当IoU < target_iou时攻击成功
            target_iou = orig_iou * (1.0 - kappa)  # 目标IoU降低kappa比例
            f = mean_iou - target_iou
            
            # 总损失：L2距离 + c * max(0, f)
            loss = l2_dist.mean() + c * torch.clamp(f, min=0)
            
            loss.backward()
            optimizer.step()
            
            # 投影到epsilon球内（保持扰动不可见）
            with torch.no_grad():
                delta.data = torch.clamp(delta.data, -epsilon, epsilon)
            
            # 记录最佳结果
            with torch.no_grad():
                current_l2 = l2_dist.mean().item()
                current_iou = mean_iou.item()
                
                # 如果IoU下降且L2距离合理，更新最佳攻击
                if current_iou < best_iou or (current_iou <= best_iou and current_l2 < best_l2):
                    best_iou = current_iou
                    best_attack = adv_images.clone()
                    best_l2 = current_l2
                
                if iteration % 20 == 0:
                    iou_drop = orig_iou - current_iou
                    perturbation_max = actual_perturbation.abs().max().item()
                    print(f"[经典C&W] Iter {iteration}: L2={current_l2:.4f}, IoU={current_iou:.4f} (下降:{iou_drop:.4f}), "
                          f"最大扰动={perturbation_max:.6f}, f={f.item():.4f}")
        
        final_iou_drop = orig_iou - best_iou
        print(f"[经典C&W] 完成! 最佳IoU下降: {final_iou_drop:.4f}, L2距离: {best_l2:.6f}")
        
        # 计算扰动统计
        with torch.no_grad():
            final_perturbation = best_attack - original_images
            max_diff = final_perturbation.abs().max().item()
            mean_diff = final_perturbation.abs().mean().item()
            
            # 检查扰动是否在可接受范围内
            if max_diff <= epsilon * 1.1:  # 允许小的数值误差
                print(f"[经典C&W] ✅ 扰动在可接受范围内 - 最大: {max_diff:.6f} (限制: {epsilon:.6f}), 平均: {mean_diff:.6f}")
            else:
                print(f"[经典C&W] ⚠️  扰动超出范围 - 最大: {max_diff:.6f} (限制: {epsilon:.6f}), 平均: {mean_diff:.6f}")
        
        return best_attack.squeeze(0)

    def apply_random_attack(self, images, rots, trans, intrins, post_rots, post_trans, binimgs,
                           intensity) -> torch.Tensor:
        """
        应用经典随机攻击，简化版本
        """
        # 经典攻击的配置
        configs = {
            'low': {
                'method': 'fgsm',
                'epsilon': 1/255,
            },
            'medium': {
                'method': 'pgd',
                'epsilon': 2/255,
                'alpha': 0.5/255,
                'steps': 10
            },
            'high': {
                'method': 'cw',
                'c': 0.5,
                'max_iter': 50
            }
        }

        if intensity not in configs:
            raise ValueError(f"不支持的强度 '{intensity}'")

        cfg = configs[intensity]
        print(f"[经典攻击配置] 强度: {intensity}, 方法: {cfg['method']}")

        if cfg['method'] == 'fgsm':
            return self.classic_fgsm_attack(images, rots, trans, intrins, post_rots, post_trans, binimgs, 
                                          epsilon=cfg['epsilon'])
        elif cfg['method'] == 'pgd':
            return self.classic_pgd_attack(images, rots, trans, intrins, post_rots, post_trans, binimgs,
                                         epsilon=cfg['epsilon'], alpha=cfg['alpha'], num_steps=cfg['steps'])
        elif cfg['method'] == 'cw':
            return self.c_w_attack(images, rots, trans, intrins, post_rots, post_trans, binimgs,
                                 c=cfg['c'], max_iter=cfg['max_iter'])
        else:
            raise ValueError(f"未知的攻击方法: {cfg['method']}") 


    def apply_fgsm_attack(self, images, rots, trans, intrins, post_rots, post_trans, binimgs,
                           intensity='medium'):
        """
        应用FGSM攻击
        """
        # 根据强度设置参数
        epsilon_configs = {
            'low': 1/255,
            'medium': 2/255, 
            'high': 4/255
        }
        epsilon = epsilon_configs.get(intensity, 2/255)
        return self.classic_fgsm_attack(images, rots, trans, intrins, post_rots, post_trans, binimgs, 
                                        epsilon=epsilon)
        
    def apply_pgd_attack(self, images, rots, trans, intrins, post_rots, post_trans, binimgs,
                          intensity='medium'):
        """
        应用PGD攻击
        """
        # 根据强度设置参数
        configs = {
            'low': {'epsilon': 1/255, 'alpha': 0.3/255, 'num_steps': 5},
            'medium': {'epsilon': 2/255, 'alpha': 0.5/255, 'num_steps': 10},
            'high': {'epsilon': 4/255, 'alpha': 1/255, 'num_steps': 15}
        }
        cfg = configs.get(intensity, configs['medium'])
        return self.classic_pgd_attack(images, rots, trans, intrins, post_rots, post_trans, binimgs, 
                                       epsilon=cfg['epsilon'], alpha=cfg['alpha'], num_steps=cfg['num_steps'])
        
    def apply_cw_attack(self, images, rots, trans, intrins, post_rots, post_trans, binimgs, 
                        intensity='medium'):
        """
        应用C&W攻击
        """
        # 根据强度设置参数
        configs = {
            'low': {'c': 0.5, 'kappa': 0.1, 'max_iter': 50, 'epsilon': 4/255},
            'medium': {'c': 1.0, 'kappa': 0.2, 'max_iter': 80, 'epsilon': 8/255},
            'high': {'c': 2.0, 'kappa': 0.3, 'max_iter': 100, 'epsilon': 16/255}
        }
        cfg = configs.get(intensity, configs['medium'])
        return self.c_w_attack(images, rots, trans, intrins, post_rots, post_trans, binimgs, 
                               c=cfg['c'], kappa=cfg['kappa'], max_iter=cfg['max_iter'], epsilon=cfg['epsilon'])
        