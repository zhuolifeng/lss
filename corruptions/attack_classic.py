import torch
import torch.nn.functional as F
import random
from typing import Optional, Callable


class ClassicAdversarialAttack:
    """
    经典版本的对抗攻击实现 - FGSM和PGD的原始版本
    """
    def __init__(self, model):
        self.model = model
        # 常见模型（如 ImageNet）使用的标准化参数
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)

    def normalize(self, images: torch.Tensor) -> torch.Tensor:
        """将 [0,255] 图像归一化到模型所需的分布"""
        images = images.float() / 255.0
        return (images - self.mean.to(images.device)) / self.std.to(images.device)

    def denormalize(self, images: torch.Tensor) -> torch.Tensor:
        """从模型分布反归一化回 [0,1] 再转 0~255"""
        images = images * self.std.to(images.device) + self.mean.to(images.device)
        images = torch.clamp(images, 0, 1)
        return (images * 255.0).round().byte()

    def denormalize_image(self, image: torch.Tensor) -> torch.Tensor:
        """将[0,1]范围的图像转换回[0,255]整数格式"""
        return torch.clamp(image * 255, 0, 255).byte()

    def classic_fgsm_attack(self, images, rots, trans, intrins, post_rots, post_trans, binimgs,
                           epsilon=2/255):
        """
        经典FGSM攻击实现 - 增强版
        使用组合损失来更有效地降低IoU
        """
        print(f"[经典FGSM] epsilon: {epsilon:.6f} (像素级: {epsilon*255:.1f}/255)")
        
        # 转换epsilon到归一化空间 - 使用更精确的转换
        eps_normalized = epsilon / torch.tensor([0.229, 0.224, 0.225]).to(images.device).view(1, 3, 1, 1)
        
        # 输入images已经是归一化后的张量
        images_tensor = images.unsqueeze(0).clone().detach().requires_grad_(True)

        # 前向传播
        output = self.model(images_tensor, rots.unsqueeze(0), trans.unsqueeze(0),
                           intrins.unsqueeze(0), post_rots.unsqueeze(0), post_trans.unsqueeze(0))
        
        # 使用更有效的损失组合
        target = binimgs.unsqueeze(0).float()
        
        # 1. 标准二元交叉熵损失
        loss_ce = F.binary_cross_entropy_with_logits(output, target)
        
        # 2. 直接的IoU损失 - 这是关键！
        pred = output.sigmoid()
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection
        iou = intersection / (union + 1e-8)
        loss_iou = iou.mean()  # 我们想最大化这个损失（最小化IoU）
        
        # 3. 针对车辆区域的抑制
        vehicle_mask = target
        loss_suppress = (pred * vehicle_mask).mean()
        
        # 组合损失 - 主要依靠IoU损失
        total_loss = loss_ce - 2.0 * loss_iou + loss_suppress
        
        # 计算原始IoU
        with torch.no_grad():
            orig_iou = iou.mean().item()
            print(f"[经典FGSM] 原始IoU: {orig_iou:.4f}")
        
        # 计算梯度
        total_loss.backward()
        
        # FGSM核心：epsilon * sign(gradient)
        data_grad = images_tensor.grad.data
        sign_data_grad = data_grad.sign()
        
        # 生成对抗样本
        perturbed_image = images_tensor + eps_normalized * sign_data_grad
        
        # 简单裁剪到合理范围（经典版本的唯一约束）
        perturbed_image = torch.clamp(perturbed_image, 
                                    images_tensor - eps_normalized, 
                                    images_tensor + eps_normalized)
        
        # 计算攻击后的IoU
        with torch.no_grad():
            attacked_output = self.model(perturbed_image, rots.unsqueeze(0), trans.unsqueeze(0),
                                       intrins.unsqueeze(0), post_rots.unsqueeze(0), post_trans.unsqueeze(0))
            attacked_pred = attacked_output.sigmoid()
            attacked_intersection = (attacked_pred * target).sum(dim=(2, 3))
            attacked_union = attacked_pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - attacked_intersection
            attacked_iou = (attacked_intersection / (attacked_union + 1e-8)).mean().item()
            print(f"[经典FGSM] 攻击后IoU: {attacked_iou:.4f} (下降: {orig_iou - attacked_iou:.4f})")
        
        # 计算扰动统计
        diff = (perturbed_image - images_tensor).abs()
        max_diff = diff.max()
        mean_diff = diff.mean()
        
        # 计算像素级差异（转换到[0,1]空间）
        orig_img_01 = images_tensor * self.std.to(images_tensor.device) + self.mean.to(images_tensor.device)
        adv_img_01 = perturbed_image * self.std.to(perturbed_image.device) + self.mean.to(perturbed_image.device)
        pixel_diff = (adv_img_01 - orig_img_01).abs() * 255
        max_pixel_diff = pixel_diff.max()
        mean_pixel_diff = pixel_diff.mean()
        
        print(f"[经典FGSM] 归一化空间扰动 - 最大: {max_diff:.6f}, 平均: {mean_diff:.6f}")
        print(f"[经典FGSM] 像素级差异 - 最大: {max_pixel_diff:.2f}/255 ({max_pixel_diff/255*100:.2f}%), 平均: {mean_pixel_diff:.2f}/255")
        
        return perturbed_image.squeeze(0)

    def classic_pgd_attack(self, images, rots, trans, intrins, post_rots, post_trans, binimgs,
                          epsilon=2/255, alpha=0.5/255, num_steps=10):
        """
        经典PGD攻击实现 - 增强版
        多步迭代攻击，使用IoU导向的损失
        """
        print(f"[经典PGD] epsilon: {epsilon:.6f}, alpha: {alpha:.6f}, steps: {num_steps}")
        
        # 转换到归一化空间 - 使用更精确的转换
        eps_normalized = epsilon / torch.tensor([0.229, 0.224, 0.225]).to(images.device).view(1, 3, 1, 1)
        alpha_normalized = alpha / torch.tensor([0.229, 0.224, 0.225]).to(images.device).view(1, 3, 1, 1)
        
        images_tensor = images.unsqueeze(0).clone().detach()
        target = binimgs.unsqueeze(0).float()
        
        # 随机初始化扰动
        delta = torch.zeros_like(images_tensor).uniform_(-epsilon, epsilon) 
        delta = delta * eps_normalized / epsilon  # 调整到归一化空间
        delta.requires_grad_(True)
        
        # 计算原始IoU
        with torch.no_grad():
            original_output = self.model(images_tensor, rots.unsqueeze(0), trans.unsqueeze(0),
                                       intrins.unsqueeze(0), post_rots.unsqueeze(0), post_trans.unsqueeze(0))
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
            
            # 前向传播
            output = self.model(images_tensor + delta, rots.unsqueeze(0), trans.unsqueeze(0),
                               intrins.unsqueeze(0), post_rots.unsqueeze(0), post_trans.unsqueeze(0))
            
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
            
            # 反向传播
            total_loss.backward()
            
            # PGD更新：delta = delta + alpha * sign(gradient)
            delta.data = delta.data + alpha_normalized * delta.grad.sign()
            
            # 投影到L∞球内
            delta.data = torch.clamp(delta.data, -eps_normalized, eps_normalized)
            
            # 确保攻击后的图像在合理范围内
            delta.data = torch.clamp(images_tensor + delta.data, 0, 1) - images_tensor
            
            if step % 3 == 0:
                current_iou = iou.mean().item()
                print(f"[经典PGD] Step {step}: IoU = {current_iou:.4f}")
        
        # 生成最终对抗样本
        perturbed_image = images_tensor + delta
        
        # 计算最终IoU
        with torch.no_grad():
            final_output = self.model(perturbed_image, rots.unsqueeze(0), trans.unsqueeze(0),
                                    intrins.unsqueeze(0), post_rots.unsqueeze(0), post_trans.unsqueeze(0))
            final_pred = final_output.sigmoid()
            final_intersection = (final_pred * target).sum(dim=(2, 3))
            final_union = final_pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - final_intersection
            final_iou = (final_intersection / (final_union + 1e-8)).mean().item()
            print(f"[经典PGD] 最终IoU: {final_iou:.4f} (下降: {orig_iou - final_iou:.4f})")
        
        # 扰动统计
        diff = delta.abs()
        max_diff = diff.max()
        mean_diff = diff.mean()
        
        # 计算像素级差异（转换到[0,1]空间）
        orig_img_01 = images_tensor * self.std.to(images_tensor.device) + self.mean.to(images_tensor.device)
        adv_img_01 = perturbed_image * self.std.to(perturbed_image.device) + self.mean.to(perturbed_image.device)
        pixel_diff = (adv_img_01 - orig_img_01).abs() * 255
        max_pixel_diff = pixel_diff.max()
        mean_pixel_diff = pixel_diff.mean()
        
        print(f"[经典PGD] 归一化空间扰动 - 最大: {max_diff:.6f}, 平均: {mean_diff:.6f}")
        print(f"[经典PGD] 像素级差异 - 最大: {max_pixel_diff:.2f}/255 ({max_pixel_diff/255*100:.2f}%), 平均: {mean_pixel_diff:.2f}/255")
        
        return perturbed_image.squeeze(0)

    def c_w_attack(self, images, rots, trans, intrins, post_rots, post_trans, binimgs,
                   c=1.0, kappa=0, max_iter=100, learning_rate=0.01):
        """
        经典C&W攻击实现
        使用L2范数约束的优化攻击
        """
        print(f"[经典C&W] c: {c}, kappa: {kappa}, max_iter: {max_iter}")
        
        images_tensor = images.unsqueeze(0).clone().detach()
        batch_size = images_tensor.size(0)
        
        # 初始化扰动变量w，使用arctanh变换确保图像在[0,1]范围内
        w = torch.zeros_like(images_tensor, requires_grad=True)
        optimizer = torch.optim.Adam([w], lr=learning_rate)
        
        # 将图像转换到arctanh空间
        def to_tanh_space(x):
            return torch.atanh((x - 0.5) * 1.99999)  # 避免arctanh的数值问题
        
        def from_tanh_space(x):
            return torch.tanh(x) * 0.5 + 0.5
        
        original_images_tanh = to_tanh_space(images_tensor)
        
        best_l2 = float('inf')
        best_attack = images_tensor.clone()
        
        print(f"[经典C&W] 开始优化...")
        
        for iteration in range(max_iter):
            optimizer.zero_grad()
            
            # 从tanh空间生成对抗样本
            adv_images = from_tanh_space(original_images_tanh + w)
            
            # 前向传播
            outputs = self.model(adv_images, rots.unsqueeze(0), trans.unsqueeze(0),
                                intrins.unsqueeze(0), post_rots.unsqueeze(0), post_trans.unsqueeze(0))
            
            # C&W损失函数
            # L2距离项
            l2_dist = torch.norm((adv_images - images_tensor).view(batch_size, -1), dim=1)
            
            # 攻击成功项 (对于分割任务，我们希望降低IoU)
            probs = torch.sigmoid(outputs)
            target_probs = binimgs.unsqueeze(0).float()
            
            # 计算IoU损失（希望最小化IoU）
            intersection = (probs * target_probs).sum(dim=(2, 3))
            union = probs.sum(dim=(2, 3)) + target_probs.sum(dim=(2, 3)) - intersection
            iou = intersection / (union + 1e-8)
            
            # C&W的f函数：我们希望攻击成功（IoU降低）
            f = iou.mean() - kappa  # 当IoU < kappa时攻击成功
            
            # 总损失
            loss = l2_dist.mean() + c * torch.max(torch.tensor(0.0, device=f.device), f)
            
            loss.backward()
            optimizer.step()
            
            # 记录最佳结果
            current_l2 = l2_dist.mean().item()
            if current_l2 < best_l2 and f.item() <= 0:
                best_l2 = current_l2
                best_attack = adv_images.clone()
            
            if iteration % 20 == 0:
                print(f"[经典C&W] Iter {iteration}: L2={current_l2:.4f}, f={f.item():.4f}, IoU={iou.mean().item():.4f}")
        
        print(f"[经典C&W] 完成! 最佳L2距离: {best_l2:.6f}")
        
        return best_attack.squeeze(0)

    def apply_random_attack(self, images, rots, trans, intrins, post_rots, post_trans, binimgs,
                           intensity) -> torch.Tensor:
        """
        应用经典随机攻击，保持与corrupt.py的接口一致
        """
        # 经典攻击的配置（使用更小的epsilon值来减少可见性）
        configs = {
            'low': {
                'method': 'fgsm',
                'epsilon': 1/255,  # 更小的扰动
            },
            'medium': {
                'method': 'pgd',
                'epsilon': 2/255,  # 中等扰动
                'alpha': 0.5/255,  # 更小的步长
                'steps': 10
            },
            'high': {
                'method': 'cw',
                'c': 0.5,  # 减小C&W的c参数
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