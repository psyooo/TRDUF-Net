# -*- coding: utf-8 -*-
"""
❗❗❗❗❗❗#此py作用：第三阶段的决策融合 (适配 TR-DUN 丰度引导版 + 可视化)
"""
import torch
from .evaluation1 import MetricsCal
import os
import scipy.io as sio
import numpy as np
from datetime import datetime
import torch.nn.functional as F
import torch.nn as nn
from torchmetrics.functional import structural_similarity_index_measure as ssim
from .visualizer import UnifiedVisualizer


def spatial_spectral_total_variation(image, spatial_weight=0.5, spectral_weight=0.5):
    """
    计算空间光谱总变分 (SSTV)
    :param image: 输入图像，形状为 (B, C, H, W)
    """
    # 空间维度梯度 (TV)
    dx = torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])
    dy = torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :])
    spatial_tv = torch.sum(dx) + torch.sum(dy)

    # 光谱维度梯度 (Spectral TV)
    db = torch.abs(image[:, :-1, :, :] - image[:, 1:, :, :])
    spectral_tv = torch.sum(db)

    return spatial_weight * spatial_tv + spectral_weight * spectral_tv


class TR_Fusion_Select:
    """
    针对 TR-DUN 输出的重构图进行最终的决策微调
    利用丰度图作为空间结构权重引导，保护物质边界，并集成可视化
    """

    def __init__(self, args, tr_hsi, tr_abundance):
        self.args = args
        self.device = args.device

        # 初始化可视化工具
        self.vis = UnifiedVisualizer(
            env_name=f"{args.data_name}_Stage3_Fusion",
            save_dir=os.path.join(args.expr_dir, "visualizations_s3")
        )

        # 获取 TR-DUN 的输出并脱离计算图作为优化基准
        self.tr_hsi = tr_hsi.detach()
        self.tr_abundance = tr_abundance.detach()

        # 待优化的参数：初始化为 TR-DUN 的输出
        self.fused_res = nn.Parameter(self.tr_hsi.clone())
        self.optimizer = torch.optim.Adam([self.fused_res], lr=1e-4)

        # 损失窗口标识
        self.loss_win = 'Stage3_Total_Loss'

    def train(self, gt_np):
        print("🚀 开始第三阶段：基于丰度引导的决策融合微调...")

        # 1. 生成引导权重图
        with torch.no_grad():
            # 取所有端元通道的最大响应作为结构参考
            weight_map = torch.max(self.tr_abundance, dim=1, keepdim=True)[0]

            if weight_map.shape[2:] != self.tr_hsi.shape[2:]:
                weight_map = F.interpolate(weight_map, size=self.tr_hsi.shape[2:], mode='bilinear', align_corners=True)

            # 归一化权重到 [0.1, 1.0]
            weight_map = (weight_map - weight_map.min()) / (weight_map.max() - weight_map.min() + 1e-8)
            weight_map = weight_map * 0.9 + 0.1
            self.weight_map_for_vis = weight_map  # 留作可视化

        # 2. 开始微调循环
        for iter in range(200):
            self.optimizer.zero_grad()

            # Loss A: 物理忠实度损失
            loss_fidelity = F.l1_loss(self.fused_res, self.tr_hsi)

            # Loss B: 丰度引导的 SSTV
            loss_sstv = spatial_spectral_total_variation(self.fused_res * weight_map)

            total_loss = loss_fidelity + 0.005 * loss_sstv

            total_loss.backward()
            self.optimizer.step()

            # 实时可视化 Loss
            if (iter + 1) % 10 == 0:
                self.vis.vis.line(
                    X=np.array([iter + 1]),
                    Y=np.array([total_loss.item()]),
                    win=self.loss_win,
                    update='append' if iter > 0 else None,
                    opts=dict(title='Stage 3 Fusion Loss', xlabel='Iteration', ylabel='Loss', legend=['Total Loss'])
                )

            if (iter + 1) % 50 == 0:
                with torch.no_grad():
                    res_np = self.fused_res.detach().cpu().numpy()[0].transpose(1, 2, 0)
                    res_np = np.clip(res_np, 0, 1)
                    metrics = MetricsCal(gt_np, res_np, self.args.scale_factor)
                    print(f"融合迭代 {iter + 1}/200 | PSNR: {metrics[0]:.4f} | Loss: {total_loss.item():.6f}")

                    # 更新 Visdom 中的重构图像预览
                    self.vis.display_recon(self.fused_res, title=f"S3_Iter_{iter + 1}", win_key='fused')

        # 3. 保存并最终对比可视化
        self.save_final_mat()
        self.final_visualize(gt_np)

        return self.fused_res.detach()

    def final_visualize(self, gt_np):
        """最终结果的综合可视化对比"""
        with torch.no_grad():
            res_tensor = self.fused_res.detach()
            gt_tensor = torch.from_numpy(gt_np.transpose(2, 0, 1)).unsqueeze(0).to(self.device).float()

            # 调用可视化器的双栏对比功能
            # 左侧显示 GT，右侧显示融合后的重构结果
            self.vis.display_results(
                gt_tensor,
                res_tensor,
                img_name="Stage3_Final_Comparison",
                left_title="Ground Truth",
                right_title="Fused Result (TR-DUN)"
            )

            # 额外可视化权重引导图（显著性图）
            self.vis.display_results(
                self.weight_map_for_vis,
                self.weight_map_for_vis,
                img_name="Abundance_Weight_Map",
                left_title="Weight Map (Heatmap)",
                right_title="Weight Map (Gray)",
                is_heatmap=True
            )

    def save_final_mat(self):
        """将最终的融合结果保存为 .mat 文件"""
        final_out = self.fused_res.detach().cpu().numpy()[0].transpose(1, 2, 0)
        save_path = os.path.join(self.args.expr_dir, 'final_fused_result.mat')
        sio.savemat(save_path, {'Out': final_out})
        print(f"✅ 最终融合结果及可视化已导出至: {self.args.expr_dir}")


def run_stage3(args, tr_hsi, tr_abundance, gt_np):
    """
    提供给 main.py 调用的接口函数
    """
    selector = TR_Fusion_Select(args, tr_hsi, tr_abundance)
    final_res = selector.train(gt_np)
    return final_res