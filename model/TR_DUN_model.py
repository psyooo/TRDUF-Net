# -*- coding: utf-8 -*-
"""
TR-DUN: Tensor Ring Deep Unfolding Network
用于高光谱与多光谱图像融合（HSI-MSI Fusion）
该模型将物理退化模型与深度先验（液态注意力机制）相结合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ============================================================
# 1. 液态光谱注意力机制 (Liquid Spectral Attention, LSA)
# 用于捕获丰度图中的光谱/端元间的动态演化特征
# ============================================================
class LiquidSpectralAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.ch = channels
        # 液态神经元动力学参数
        self.weight = nn.Parameter(torch.randn(channels, channels) * 0.001)
        self.bias = nn.Parameter(torch.zeros(channels))
        self.tau = nn.Parameter(torch.ones(1, channels, 1, 1) * 2.0)  # 时间常数 τ

    def forward(self, x):
        B, C, H, W = x.shape
        # 1. 全局统计描述符 (Global Average Pooling)
        s = torch.mean(x, dim=(2, 3), keepdim=True)  # [B, C, 1, 1]

        # 2. 液态神经元动力学计算
        dt = 1.0
        # 将 tau 映射到正数区间 [0.1, 4.1]
        tau = torch.sigmoid(self.tau) * 4.0 + 0.1
        alpha = torch.exp(-dt / tau)

        # 状态更新演化 (类似于 RNN 的隐状态更新)
        hidden = torch.tanh(s)
        out = alpha * s + (1 - alpha) * hidden

        # 3. 输出注意力权重并作用于输入特征
        attention_weight = torch.sigmoid(out)
        return x * attention_weight


# ============================================================
# 2. TR-Prox 模块 (张量环先验近端映射)
# 负责在每一阶段学习丰度图 A 的深层先验分布
# ============================================================
class TRProxBlock(nn.Module):
    def __init__(self, abundance_dim):
        super(TRProxBlock, self).__init__()
        # 使用残差卷积结构结合液态注意力
        self.net = nn.Sequential(
            nn.Conv2d(abundance_dim, abundance_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(abundance_dim * 2),
            nn.ReLU(inplace=True),
            LiquidSpectralAttention(abundance_dim * 2),
            nn.Conv2d(abundance_dim * 2, abundance_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(abundance_dim)
        )

    def forward(self, a_in):
        # A_out = A_in + Network(A_in)
        return a_in + self.net(a_in)


# ============================================================
# 3. TRDUNStage (单阶段迭代单元)
# 实现 X 和 A 的交替方向乘子法 (ADMM) 或近端梯度下降 (PGD) 步
# ============================================================
class TRDUNStage(nn.Module):
    def __init__(self, hs_bands, ms_bands, abundance_dim):
        super(TRDUNStage, self).__init__()
        # 物理演化步长参数（可学习）
        self.eta = nn.Parameter(torch.Tensor([0.1]))  # X 更新权重
        self.mu = nn.Parameter(torch.Tensor([0.1]))  # A 更新权重
        self.rho = nn.Parameter(torch.Tensor([0.05]))  # 解混耦合系数

        # 本阶段先验网络
        self.prox_net = TRProxBlock(abundance_dim)

    def forward(self, x, a, y_h, y_m, e, psf, srf, ratio):
        """
        x: HR-HSI [B, L, H, W]
        a: Abundance [B, R, H, W]
        y_h: LR-HSI, y_m: HR-MSI, e: Endmembers, psf/srf: Kernels
        """
        B, L, H, W = x.shape

        # --- 步骤 A: 数据保真项更新 (Update X) ---
        # 1. 空间退化残差 (模糊+下采样)
        pad = psf.shape[-1] // 2
        psf_stack = psf.repeat(L, 1, 1, 1)  # 为组卷积准备
        x_blur_down = F.conv2d(x, psf_stack, stride=ratio, groups=L)
        res_h = x_blur_down - y_h
        # 空间反向投影 (梯度算子)
        grad_x_h = F.interpolate(res_h, size=(H, W), mode='bilinear', align_corners=True)

        # 2. 光谱退化残差 (光谱响应)
        x_spec = torch.einsum('cl, blhw -> bchw', srf, x)
        res_m = x_spec - y_m
        # 光谱反向投影
        grad_x_m = torch.einsum('cl, bchw -> blhw', srf, res_m)

        # 3. 线性解混耦合项 (X - EA)
        ea = torch.einsum('lr, brhw -> blhw', e, a)
        grad_coupling = x - ea

        # 更新 X
        x_next = x - self.eta * (grad_x_h + grad_x_m + self.rho * grad_coupling)

        # --- 步骤 B: 丰度图更新 (Update A) ---
        # 梯度更新：E^T * (EA - X_next)
        res_a = ea - x_next
        grad_a = torch.einsum('lr, blhw -> brhw', e, res_a)

        # 深度先验投影 (Proximal Mapping)
        v = a - self.mu * self.rho * grad_a
        a_prox = self.prox_net(v)

        # 物理约束：非负性 (ANC) 与 归一化 (ASC)
        a_next = F.relu(a_prox)
        a_sum = torch.sum(a_next, dim=1, keepdim=True) + 1e-8
        a_next = a_next / a_sum

        return x_next, a_next


# ============================================================
# 4. TRDUN (主网络架构)
# ============================================================
class TRDUN(nn.Module):
    def __init__(self, hs_bands, ms_bands, abundance_dim=40, stages=5):
        super(TRDUN, self).__init__()
        self.abundance_dim = abundance_dim
        self.stages_count = stages

        # 1. 全局端元矩阵 E (Learnable Dictionary)
        # 对应线性解混模型 X = EA
        self.E_matrix = nn.Parameter(torch.randn(hs_bands, abundance_dim) * 0.1)

        # 2. 级联阶段
        self.unfold_stages = nn.ModuleList([
            TRDUNStage(hs_bands, ms_bands, abundance_dim)
            for _ in range(stages)
        ])

        # 3. 初始化卷积 (用于生成 X_0 和 A_0)
        self.init_x = nn.Sequential(
            nn.Conv2d(hs_bands, hs_bands, 3, padding=1),
            nn.PReLU()
        )
        self.init_a = nn.Sequential(
            nn.Conv2d(hs_bands, abundance_dim, 3, padding=1),
            nn.Softmax(dim=1)
        )

    def forward(self, y_h, y_m, psf, srf, ratio):
        # 确保端元非负
        e_pos = F.relu(self.E_matrix)

        # 初始化：通过双线性插值获取初始高分辨率估计
        x = F.interpolate(y_h, size=y_m.shape[2:], mode='bilinear', align_corners=True)
        x = self.init_x(x) + x

        # 初始丰度图
        a = self.init_a(x)

        # 迭代演化
        x_list = []
        for stage in self.unfold_stages:
            x, a = stage(x, a, y_h, y_m, e_pos, psf, srf, ratio)
            x_list.append(x)  # 记录每一层输出用于辅助 Loss

        return x, a, x_list