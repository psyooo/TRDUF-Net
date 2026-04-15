import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import os
import scipy.io as sio
import numpy as np
import torch.nn.functional as F

from .read_data import readdata
from .evaluation import MetricsCal
from .TR_DUN_model import TRDUN


class TR_Reconstruction:
    def __init__(self, args, psf_prior, srf_prior):
        self.args = args
        self.data = readdata(self.args)

        # 从 read_data 获取观测数据并转为 GPU Tensor
        self.y_h = torch.Tensor(self.data.lr_hsi).unsqueeze(0).to(self.args.device)
        self.y_m = torch.Tensor(self.data.hr_msi).unsqueeze(0).to(self.args.device)
        self.gt = self.data.gt  # [H, W, C] numpy 格式真实值，用于评估

        # 接收第一阶段提取的物理先验
        self.psf = psf_prior.to(self.args.device)
        self.srf = srf_prior.to(self.args.device)

        hs_bands = self.y_h.shape[1]
        ms_bands = self.y_m.shape[1]

        # 初始化 TR-DUN 网络 (Rank 直接作为端元维数 abundance_dim)
        self.net = TRDUN(hs_bands=hs_bands, ms_bands=ms_bands,
                         abundance_dim=self.args.Rank, stages=5).to(self.args.device)

        # 优化器与学习率衰减
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.args.lr_stage2_UNet)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=self.args.niter_decay2_UNet, gamma=0.5)

    def train(self):
        print("🚀 开始第二阶段：TR-DUN 深度展开网络训练...")
        best_psnr = 0.0
        best_out = None
        best_a = None

        for epoch in range(self.args.niter2_UNet):
            self.net.train()
            self.optimizer.zero_grad()

            # 前向传播
            x_out, a_out, x_list = self.net(self.y_h, self.y_m, self.psf, self.srf, self.args.scale_factor)

            # ==========================
            # 级联损失函数 (多阶段物理保真)
            # ==========================
            loss = 0
            # 1. 约束中间级联过程的物理保真
            for x_stage in x_list:
                # 空间模糊保真
                pad = self.psf.shape[-1] // 2
                x_blur = F.conv2d(x_stage, self.psf.repeat(x_stage.shape[1], 1, 1, 1),
                                  stride=self.args.scale_factor, groups=x_stage.shape[1], padding=pad)
                loss += F.l1_loss(x_blur, self.y_h)

                # 光谱投影保真
                x_spec = torch.einsum('cl, blhw -> bchw', self.srf, x_stage)
                loss += F.l1_loss(x_spec, self.y_m)

            # 2. 最终输出的强化约束
            pad = self.psf.shape[-1] // 2
            x_out_blur = F.conv2d(x_out, self.psf.repeat(x_out.shape[1], 1, 1, 1),
                                  stride=self.args.scale_factor, groups=x_out.shape[1], padding=pad)
            loss_h = F.l1_loss(x_out_blur, self.y_h)

            x_out_spec = torch.einsum('cl, blhw -> bchw', self.srf, x_out)
            loss_m = F.l1_loss(x_out_spec, self.y_m)

            # 最终 Loss 组合
            total_loss = loss + 2.0 * (loss_h + loss_m)

            total_loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # 定期评估与保存最佳结果
            if (epoch + 1) % 10 == 0 or epoch == 0:
                self.net.eval()
                with torch.no_grad():
                    out_np = x_out.squeeze(0).cpu().numpy().transpose(1, 2, 0)
                    out_np = np.clip(out_np, 0, 1)

                    # 调用你们原本的 MetricsCal 计算精度
                    metrics = MetricsCal(self.gt, out_np, self.args.scale_factor)
                    psnr = metrics[0] if isinstance(metrics, (tuple, list)) else metrics

                    print(
                        f"Epoch: [{epoch + 1}/{self.args.niter2_UNet}] | Loss: {total_loss.item():.4f} | PSNR: {psnr}")

                    # 最优模型保存逻辑
                    current_psnr = psnr if isinstance(psnr, float) else float(str(psnr).split(',')[0].strip('()'))
                    if current_psnr > best_psnr:
                        best_psnr = current_psnr
                        best_out = x_out.clone().detach()
                        best_a = a_out.clone().detach()

        # 训练结束，保存结果
        if not os.path.exists(self.args.expr_dir):
            os.makedirs(self.args.expr_dir)
        sio.savemat(os.path.join(self.args.expr_dir, 'hrhsi_TRDUN.mat'),
                    {'Out': best_out.cpu().numpy()[0].transpose(1, 2, 0)})
        sio.savemat(os.path.join(self.args.expr_dir, 'abundance_TRDUN.mat'),
                    {'Abundance': best_a.cpu().numpy()[0].transpose(1, 2, 0)})

        print(f"🎉 TR-DUN 第二阶段完成！最佳 PSNR: {best_psnr:.4f}")
        return best_out, best_a