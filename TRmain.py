# -*- coding: utf-8 -*-
"""
❗❗❗❗❗❗#主文件 (适配 TR-DUN)
"""
import os
import torch
import time
import numpy as np
import random
from model.config import args
import matplotlib.pyplot as plt
from model.visualizer import UnifiedVisualizer
from model.evaluation1 import MetricsCal
from thop import profile


# 设置随机种子
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(args.seed)

# 初始化可视化工具
vis = UnifiedVisualizer(
    env_name=f"{args.data_name}_visualization",
    save_dir=os.path.join(args.expr_dir, "visualizations")
)

'''第一阶段：退化估计 (BlindNet)'''
from model.srf_psf_layer import Blind

blind = Blind(args)

start_s1 = time.perf_counter()
lr_msi_fhsi_est, lr_msi_fmsi_est = blind.train()
blind.get_save_result()
end_s1 = time.perf_counter()
elapsed_S1 = end_s1 - start_s1

# 获取估计的物理参数
psf_est = blind.model.psf.data.detach()
srf_est = torch.squeeze(blind.model.srf.data.detach()).T  # L x C

'''第二阶段：TR-DUN 深度展开重构'''
from model.TR_DUN_model import TRDUN
from torch.optim import Adam

# 初始化模型
# hs_bands: 46, ms_bands: 8, abundance_dim: 40 (Rank)
tr_model = TRDUN(
    hs_bands=blind.tensor_gt.shape[1],
    ms_bands=blind.ms_bands,
    abundance_dim=blind.P,
    stages=5
).to(args.device)

optimizer = Adam(tr_model.parameters(), lr=args.lr_stage2_UNet)
start_s2 = time.perf_counter()

print("🚀 开始第二阶段：TR-DUN 深度展开训练...")
# 简化的训练循环 (实际可根据需要增加 epoch)
for epoch in range(args.niter2_UNet):
    optimizer.zero_grad()

    # y_h: blind.data.lr_hsi_tensor, y_m: blind.data.hr_msi_tensor
    hrhsi_recon, abundance_map, x_list = tr_model(
        blind.tensor_lr_hsi,
        blind.tensor_hr_msi,
        psf_est,
        srf_est,
        args.scale_factor
    )

    # 计算损失 (最后一层输出 + 阶段性辅助损失)
    loss_final = torch.mean(torch.abs(hrhsi_recon - blind.gt_tensor))
    loss_stages = sum([torch.mean(torch.abs(x - blind.gt_tensor)) for x in x_list])
    total_loss = loss_final + 0.1 * loss_stages

    total_loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}/{args.niter2_UNet} | Loss: {total_loss.item():.6f}")

end_s2 = time.perf_counter()
elapsed_S2 = end_s2 - start_s2

'''第三阶段：基于丰度引导的决策融合'''
from model.Dfusion import run_stage3

start_s3 = time.perf_counter()

# 传入 TR-DUN 的重构结果、丰度图和真实值(用于评估)
fused_final = run_stage3(args, hrhsi_recon, abundance_map, blind.gt)

end_s3 = time.perf_counter()
elapsed_S3 = end_s3 - start_s3

# 最终指标评估
final_np = fused_final.cpu().numpy()[0].transpose(1, 2, 0)
final_np = np.clip(final_np, 0, 1)
metrics = MetricsCal(blind.gt, final_np, args.scale_factor)

print("\n" + "=" * 30)
print(f"最终结果指标:")
print(f"PSNR: {metrics[0]:.4f}")
print(f"SAM:  {metrics[1]:.4f}")
print(f"ERGAS: {metrics[2]:.4f}")
print(f"SSIM: {metrics[3]:.4f}")
print(f"Time: S1:{elapsed_S1:.2f}s, S2:{elapsed_S2:.2f}s, S3:{elapsed_S3:.2f}s")
print("=" * 30)