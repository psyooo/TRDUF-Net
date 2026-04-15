# -*- coding: utf-8 -*-

"""
❗❗❗❗❗❗#此py作用：读取数据 并且仿真生成lrhsi和hrmsi
"""

import os
import numpy as np
import torch
import scipy.io as io
import xlrd
import cv2
from scipy import signal
import cvxpy as cvx
import torch.nn.functional as F

class readdata():
    def __init__(self, args):
        
        self.args = args
        self.srf_gt = self.get_spectral_response(self.args.data_name)    # hs_band X ms_band 的二维矩阵
        self.psf_gt = self.matlab_style_gauss2D(shape=(self.args.scale_factor,self.args.scale_factor),sigma=self.args.sigma)    # ratio X ratio 的二维矩阵
        self.sp_range = self.get_sp_range(self.srf_gt)
        data_folder = os.path.join(self.args.default_datapath, args.data_name)
        if os.path.exists(data_folder):
            data_path = os.path.join(data_folder, "REF.mat")
        else:
            return 0

        self.gt=io.loadmat(data_path)['REF']

        'low HSI'
        self.lr_hsi=self.generate_low_HSI(self.gt, self.args.scale_factor)
        
        'high MSI'
        self.hr_msi=self.generate_MSI(self.gt, self.srf_gt)
        
        '从msi空间降采样'
        self.lr_msi_fmsi = self.generate_low_HSI(self.hr_msi, self.args.scale_factor)
        
        '从lrhsi光谱降采样'
        self.lr_msi_fhsi= self.generate_MSI(self.lr_hsi, self.srf_gt)

        '判断是否增加噪声'
        if args.noise == 'Yes':
            
            sigmam_hsi=np.sqrt(   (self.lr_hsi**2).sum() / (10**(args.nSNR/10)) / (self.lr_hsi.size) )
            t=np.random.randn(self.lr_hsi.shape[0],self.lr_hsi.shape[1],self.lr_hsi.shape[2])
            self.lr_hsi=self.lr_hsi+sigmam_hsi*t
        
            sigmam_msi=np.sqrt(   (self.hr_msi**2).sum() / (10**(args.nSNR/10)) / (self.hr_msi.size) )
            t=np.random.randn(self.hr_msi.shape[0],self.hr_msi.shape[1],self.hr_msi.shape[2])
            self.hr_msi=self.hr_msi+sigmam_msi*t

        # ======================================================================
        # 🔥 新增：自动获取初始解混先验（端元 M0 + 丰度 A0）
        # ======================================================================
        self.P, self.M0, self.A0 = self.get_initial_prior()  # 👈 直接加这一行
        self.tensor_M0 = torch.from_numpy(self.M0).unsqueeze(0).float().to(args.device)
        self.tensor_A0 = torch.from_numpy(self.A0.transpose(2, 0, 1).copy()).unsqueeze(0).float().to(args.device)

        '将W×H×C的numpy转化为1×C×W×H的tensor'
        self.tensor_gt = torch.from_numpy(self.gt.transpose(2,0,1).copy()).unsqueeze(0).float().to(args.device) 
        self.tensor_lr_hsi = torch.from_numpy(self.lr_hsi.transpose(2,0,1).copy()).unsqueeze(0).float().to(args.device) 
        self.tensor_hr_msi = torch.from_numpy(self.hr_msi.transpose(2,0,1).copy()).unsqueeze(0).float().to(args.device) 
        self.tensor_lr_msi_fmsi = torch.from_numpy(self.lr_msi_fmsi.transpose(2,0,1).copy()).unsqueeze(0).float().to(args.device) 
        self.tensor_lr_msi_fhsi = torch.from_numpy(self.lr_msi_fhsi.transpose(2,0,1).copy()).unsqueeze(0).float().to(args.device) 

        self.print_options()    # 保存配置
        self.save_psf_srf() # 保存真实的PSF和SRF
        self.save_lrhsi_hrmsi()  # 保存生成的lr_hsi和hr_msi
        self.save_initial_prior()  # 👈 保存先验到文件
        self.save_prior_visualization()  # 👈 保存端元图 + 丰度图
        print("readdata over")




    # ======================================================================
    # 🔥 新增：全套解混工具函数（Hysime自动选P + VCA + FCLS）
    # ======================================================================

    def hysime(self, X):
        X = X.T
        L, N = X.shape
        m = np.mean(X, axis=1, keepdims=True)
        X0 = X - m
        R = (X0 @ X0.T) / N
        U, s, _ = np.linalg.svd(R)
        s = s[s > 1e-6]
        noise = np.median(np.sqrt(np.abs(s)))
        noise = max(noise, 1e-3)
        Rn = noise ** 2 * np.eye(L)
        A = np.linalg.inv(Rn) @ R
        eigv = np.linalg.eigvalsh(A)
        P = np.sum(eigv > 1.15)
        return max(P, 3)

    def vca(self, X, p):
        """
        终极无bug版VCA - 完全解决所有维度问题
        X: [N, L]  像素数 x 波段数
        p: 端元数量
        返回 M0: [L, p]
        """
        X = X.T  # [L, N]
        L, N = X.shape
        m = np.mean(X, axis=1, keepdims=True)
        Xc = X - m

        U, _, _ = np.linalg.svd(Xc @ Xc.T / N, full_matrices=True)
        Ud = U[:, :p - 1]  # [L, p-1]

        indices = np.zeros(p, dtype=int)
        A = np.zeros((p, p))

        for i in range(p):
            w = np.zeros((p, 1))
            w[i] = 1.0
            if i > 0:
                # 【核心修复】 flatten 解决维度不匹配
                sol = np.linalg.lstsq(A[:i, :i], A[:i, i], rcond=None)[0]
                w[:i, 0] = sol.flatten()  # 关键！

            w = w / np.linalg.norm(w)

            # 计算投影方向
            if i < p - 1:
                d = Ud @ w[:p - 1]
            else:
                d = np.random.randn(L, 1)
            d = d / np.linalg.norm(d)

            # 找最优像元
            proj = d.T @ X
            indices[i] = np.argmax(np.abs(proj))

            # 更新矩阵A
            if i < p - 1:
                A[:p - 1, i] = (Ud.T @ Xc)[:, indices[i]].flatten()
            A[p - 1, i] = 1.0

        # 提取端元
        M0 = X[:, indices]
        return M0

    def fcls(self, X, M):
        """
        加速版FCLS - 无cvxpy、无循环卡顿、无维度bug
        """
        import numpy as np
        from scipy.optimize import nnls
        N, L = X.shape
        P = M.shape[1]
        A = np.zeros((N, P))

        MtM = M.T @ M
        for i in range(N):
            y = X[i]
            a, _ = nnls(MtM, M.T @ y)
            a = a / np.maximum(a.sum(), 1e-8)
            A[i] = a

        return A

        # ======================================================================
        # 🔥 新增：一键获取初始先验
        # ======================================================================

    import numpy as np
    import cv2

    def get_initial_prior(self):
        # ===================== 1. 基础配置 =====================
        H_lr, W_lr, L = self.lr_hsi.shape
        H_hr, W_hr, C_msi = self.hr_msi.shape

        # ===================== 2. LR-HSI 提取纯净端元 M0 =====================
        X_lr = self.lr_hsi.reshape(-1, L)
        P = self.hysime(X_lr)
        print(f"\n✅ 自动估计端元数量 P = {P}")

        M0 = self.vca(X_lr, P)  # [L, P]

        # ===================== 3. LR 上计算丰度（低分辨率，先做强约束） =====================
        A0_lr_flat = self.fcls(X_lr, M0)
        A0_lr = A0_lr_flat.reshape(H_lr, W_lr, P)
        # 【关键1】LR阶段先做严格物理约束，从源头避免虚高
        A0_lr = np.clip(A0_lr, 0.0, 1.0)
        sum_lr = A0_lr.sum(axis=-1, keepdims=True)
        A0_lr = A0_lr / (sum_lr + 1e-8)

        # ===================== 4. 上采样到 HR 尺寸（用更保守的插值） =====================
        # 用 INTER_LINEAR 替代 CUBIC，减少平滑溢散
        A0_hr = cv2.resize(A0_lr, (W_hr, H_hr), interpolation=cv2.INTER_LINEAR)

        # ===================== 5. 【优化滤波】收紧参数，避免过度平滑 =====================
        A0_hr_enhanced = np.zeros_like(A0_hr)
        for i in range(A0_hr.shape[-1]):
            # 【关键2】大幅收紧双边滤波参数，只锐化边缘，不扩散高值
            A0_hr_enhanced[..., i] = cv2.bilateralFilter(
                (A0_hr[..., i] * 255).astype(np.uint8),
                d=3,  # 缩小邻域，只处理局部
                sigmaColor=20,  # 大幅降低颜色权重，避免扩散
                sigmaSpace=20  # 大幅降低空间权重，避免溢散
            ) / 255.0

        A0_hr = A0_hr_enhanced

        # ===================== 6. 【终极强约束】二次物理约束+背景压制 =====================
        # 非负
        A0_hr = np.clip(A0_hr, 0.0, 1.0)
        # 背景压制：把低于阈值的丰度直接置0，彻底消除虚高背景
        A0_hr[A0_hr < 0.07] = 0.0  # 阈值可微调：0.03~0.08，越低越收紧
        # 重新做和为1约束
        sum_hr = A0_hr.sum(axis=-1, keepdims=True)
        A0_hr = A0_hr / (sum_hr + 1e-8)

        # ===================== 7. 过滤无效端元 =====================
        col_mean = A0_hr.mean(axis=(0, 1))
        valid_idx = col_mean > 7e-3
        M0 = M0[:, valid_idx]
        A0 = A0_hr[..., valid_idx]
        P_valid = M0.shape[1]

        print(f"✅ 过滤冗余端元，有效端元数量 P = {P_valid}")
        print(f"✅ 丰度已收紧，背景虚高消除，尺寸 = {A0.shape}")

        self.check_unmixing_valid(self.gt, self.lr_hsi, self.hr_msi, M0, A0)
        return P_valid, M0, A0

    # ==============================================================================
    # 🔥 一键验证：端元 + 丰度 是否提取完整（权威4项检查）
    # ==============================================================================
    def check_unmixing_valid(self,gt, lr, hr, M0, A0):
        """
        lr: [H, W, L] 原始LR-HSI
        M0: [L, P] 端元
        A0: [H, W, P] 丰度
        返回：True=完整有效 / False=不完整
        """
        H, W = hr.shape[0], hr.shape[1]
        L = lr.shape[2]
        A0_flat = A0.reshape(-1, A0.shape[-1])
        X_recon = (A0_flat @ M0.T).reshape(H, W, L)

        # --------------- 1. 重建RMSE检查 ----------------
        rmse = np.sqrt(np.mean((gt - X_recon) ** 2))
        print(f"🔍 重建 RMSE = {rmse:.6f}")

        # --------------- 2. 丰度非负检查 ----------------
        abundance_neg = (A0 < -0.001).sum()
        print(f"🔍 丰度负数数量 = {abundance_neg}")

        # --------------- 3. 丰度和为1检查 ----------------
        sum_abn = A0.sum(axis=-1)
        sum_min = sum_abn.min()
        sum_max = sum_abn.max()
        print(f"🔍 丰度和范围：[{sum_min:.3f}, {sum_max:.3f}]")

        # --------------- 4. 无效丰度图（全0/平图） ----------------
        valid_map_cnt = 0
        for i in range(A0.shape[-1]):
            m = A0[:, :, i].mean()
            std = A0[:, :, i].std()
            if m > 0.001 and std > 0.001:
                valid_map_cnt += 1
        print(f"🔍 有效丰度图数量 = {valid_map_cnt} / {A0.shape[-1]}")

        # ====================== 最终判断 ======================
        is_valid = True
        if rmse > 0.02:
            print("❌ 重建误差太大 → 提取不完整")
            is_valid = False
        if abundance_neg > 0:
            print("❌ 丰度出现负数 → 物理无效")
            is_valid = False
        if sum_min < 0.9 or sum_max > 1.1:
            print("❌ 丰度和不满足1 → 提取不完整")
            is_valid = False
        if valid_map_cnt < A0.shape[-1] * 0.5:
            print("❌ 大量丰度图无效 → 提取不完整")
            is_valid = False

        if is_valid:
            print("✅ 端元+丰度 提取完整、有效、可作为先验！")
        return is_valid

    # ==============================================================================
    # 🔥 辅助函数：拼接所有彩色丰度图为一张总图（自动适配端元数量P）
    # ==============================================================================
    def plot_merged_abundance(self, abn_data, save_path, cmap='jet'):
        import matplotlib.pyplot as plt
        import math
        P = abn_data.shape[-1]  # 有效端元数量
        # 自动计算网格行列数（尽量接近正方形，美观）
        n_cols = math.ceil(math.sqrt(P))
        n_rows = math.ceil(P / n_cols)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        axes = axes.flatten()  # 展平轴，方便循环

        # 逐个绘制丰度图
        for i in range(P):
            im = axes[i].imshow(abn_data[:, :, i], cmap=cmap)
            axes[i].set_title(f'Abundance {i + 1}', fontsize=12, pad=5)
            axes[i].axis('off')  # 关闭坐标轴
            # 为每个子图加颜色条
            cbar = fig.colorbar(im, ax=axes[i], shrink=0.8)
            cbar.ax.tick_params(labelsize=8)

        # 隐藏多余的子图（如果P不是行列数的整数倍）
        for i in range(P, n_rows * n_cols):
            axes[i].axis('off')

        fig.suptitle(f'Initial Abundance Maps (Total P={P})', fontsize=16, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # 预留标题空间
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    # ==============================================================================
    # 🔥 保存：初始端元光谱图 + 单张丰度图（灰度+彩色） + 拼接丰度总图
    # ==============================================================================
    def save_prior_visualization(self):
        import matplotlib.pyplot as plt
        # 全局配置：中文支持+负号正常显示+高清绘图
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300

        save_dir = self.args.expr_dir
        P = self.M0.shape[1]  # 过滤后的有效端元数
        L = self.M0.shape[0]
        H, W, _ = self.A0.shape

        # ==========================
        # 1. 保存 初始端元光谱曲线图 M0
        # ==========================
        plt.figure(figsize=(10, 6))
        for i in range(P):
            plt.plot(self.M0[:, i], linewidth=1.8, label=f'Endmember {i + 1}', alpha=0.8)
        plt.xlabel('Spectral Band', fontsize=12, fontweight='medium')
        plt.ylabel('Reflectance', fontsize=12, fontweight='medium')
        plt.title(f'Initial Endmember Spectra (P={P})', fontsize=14, fontweight='bold', pad=10)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.3, color='gray')
        plt.xlim(0, L - 1)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'initial_endmembers_spectra.png'), dpi=300)
        plt.close()
        print(f"✅ 初始端元光谱图已保存：initial_endmembers_spectra.png")

        # ==========================
        # 2. 保存 单张丰度图（灰度+彩色）
        # ==========================
        for i in range(P):
            abn_map = self.A0[:, :, i]
            # 灰度图（偏工程，看空间分布）
            plt.figure(figsize=(6, 6))
            plt.imshow(abn_map, cmap='gray', vmin=0, vmax=abn_map.max())
            plt.axis('off')
            plt.title(f'Abundance {i + 1} (Gray)', fontsize=12, pad=5)
            plt.savefig(os.path.join(save_dir, f'abn_gray_{i + 1}.png'), bbox_inches='tight', dpi=300)
            plt.close()
            # 彩色热力图（偏分析，看丰度大小）
            plt.figure(figsize=(6, 6))
            im = plt.imshow(abn_map, cmap='jet', vmin=0, vmax=abn_map.max())
            plt.axis('off')
            plt.title(f'Abundance {i + 1} (Color)', fontsize=12, pad=5)
            cbar = plt.colorbar(im, shrink=0.8)
            cbar.ax.tick_params(labelsize=10)
            plt.savefig(os.path.join(save_dir, f'abn_color_{i + 1}.png'), bbox_inches='tight', dpi=300)
            plt.close()
        print(f"✅ 单张丰度图（灰度+彩色）已保存：共{P}个端元")

        # ==========================
        # 3. 保存 拼接版彩色丰度总图（核心新增！）
        # ==========================
        merged_save_path = os.path.join(save_dir, f'initial_abundance_merged_P{P}.png')
        self.plot_merged_abundance(self.A0, merged_save_path, cmap='jet')
        print(f"✅ 拼接彩色丰度总图已保存：initial_abundance_merged_P{P}.png")

    # ==============================================================================
    # 🔥 保存：先验数据（mat文件，供网络读取）
    # ==============================================================================
    def save_initial_prior(self):
        save_path = os.path.join(self.args.expr_dir, 'initial_prior.mat')
        io.savemat(save_path, {
            'P': self.P,
            'M0': self.M0,  # [L, P] 端元
            'A0': self.A0  # [H, W, P] 丰度
        })
        print(f"✅ 先验数据已保存：initial_prior.mat")


    def matlab_style_gauss2D(self,shape=(3,3),sigma=2):
            m,n = [(ss-1.)/2. for ss in shape]
            y,x = np.ogrid[-m:m+1,-n:n+1]
            h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
            h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
            sumh = h.sum()
            if sumh != 0:
                h /= sumh
            return h

    def get_spectral_response(self,data_name):
        xls_path = os.path.join(self.args.sp_root_path, data_name + '.xls')
        # xls_path = os.path.join(r'E:\Code\coupled\data\spectral_response', data_name + '.xls')
        if not os.path.exists(xls_path):
            raise Exception("spectral response path does not exist")
        data = xlrd.open_workbook(xls_path)
        table = data.sheets()[0]
    
        num_cols = table.ncols
        cols_list = [np.array(table.col_values(i)).reshape(-1,1) for i in range(0,num_cols)]
    
        sp_data = np.concatenate(cols_list, axis=1)
        sp_data = sp_data / (sp_data.sum(axis=0))
    
        return sp_data   
    
    def get_sp_range(self,srf_gt):
        HSI_bands, MSI_bands = srf_gt.shape
    
        assert(HSI_bands>MSI_bands)
        sp_range = np.zeros([MSI_bands,2])
        for i in range(0,MSI_bands):
            index_dim_0, index_dim_1 = np.where(srf_gt[:,i].reshape(-1,1)>0)
            sp_range[i,0] = index_dim_0[0]      # 这是索引，不是代表具体第几个波段
            sp_range[i,1] = index_dim_0[-1]
        return sp_range
    
    def downsamplePSF(self, img,sigma,stride):
        def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
            m,n = [(ss-1.)/2. for ss in shape]
            y,x = np.ogrid[-m:m+1,-n:n+1]
            h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
            h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
            sumh = h.sum()
            if sumh != 0:
                h /= sumh
            return h
        # generate filter same with fspecial('gaussian') function
        h = matlab_style_gauss2D((stride,stride),sigma)
        if img.ndim == 3:
            img_w,img_h,img_c = img.shape
        elif img.ndim == 2:
            img_c = 1
            img_w,img_h = img.shape
            img = img.reshape((img_w,img_h,1))
        from scipy import signal
        out_img = np.zeros((img_w//(stride), img_h//(stride), img_c))
        for i in range(img_c):
            out = signal.convolve2d(img[:,:,i],h,'valid')  # signal.convolve2d 要先对卷积核顺时针旋转180°
            out_img[:,:,i] = out[::stride,::stride]
        return out_img

    def generate_low_HSI(self, img, scale_factor):
        (h, w, c) = img.shape
        img_lr = self.downsamplePSF(img, sigma=self.args.sigma, stride=scale_factor)
        return img_lr 

    def generate_MSI(self, img, srf_gt):
        w,h,c = img.shape
        self.msi_channels = srf_gt.shape[1]
        if srf_gt.shape[0] == c:
            img_msi = np.dot(img.reshape(w*h,c), srf_gt).reshape(w,h,srf_gt.shape[1])
        else:
            raise Exception("The shape of sp matrix doesnot match the image")
        return img_msi
        
    def print_options(self):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(self.args ).items()):
            comment = ''
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
       
        if not os.path.exists(self.args.expr_dir):
            os.makedirs(self.args.expr_dir)
            
        file_name = os.path.join(self.args.expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')
            
    def save_psf_srf(self):
        psf_name = os.path.join(self.args.expr_dir, 'psf_gt.mat')
        srf_name = os.path.join(self.args.expr_dir, 'srf_gt.mat')
        io.savemat(psf_name,{'psf_gt': self.psf_gt})
        io.savemat(srf_name,{'srf_gt': self.srf_gt})
        
    def save_lrhsi_hrmsi(self):
        lr_hsi_name = os.path.join(self.args.expr_dir, 'lr_hsi.mat')    # 生成的lr_hsi
        hr_msi_name = os.path.join(self.args.expr_dir, 'hr_msi.mat')    # 生成的hr_msi
        io.savemat(lr_hsi_name,{'lr_hsi': self.lr_hsi})    # 保存lr_hsi
        io.savemat(hr_msi_name,{'hr_msi': self.hr_msi})    # 保存hr_msi

        # 保存lr_msi_fhsi(从lr_hsi降采样得到的lr_msi)和lr_msi_fmsi（从hr_msi降采样得到的lr_msi）
        io.savemat(os.path.join(self.args.expr_dir , 'gt_lr_msi.mat'), {'lr_msi_fhsi_gt': self.lr_msi_fhsi, 'lr_msi_fmsi_gt': self.lr_msi_fmsi})

        
if __name__ == "__main__":
    
    
    from config import args
    im=readdata(args)