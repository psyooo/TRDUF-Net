# -*- coding: utf-8 -*-

"""
❗❗❗❗❗❗#此py作用：计算精度
"""


#说明l:输出的是W H C大小的numpy格式数据
import numpy as np

import os
import scipy.io as io


def compute_sam(x_true, x_pred):
    # print("gt.shape", x_true.shape, "gt_est_fhsi.shape", x_pred.shape)
    assert x_true.ndim ==3 and x_true.shape == x_pred.shape

    w, h, c = x_true.shape
    x_true = x_true.reshape(-1, c)
    x_pred = x_pred.reshape(-1, c)

    #sam = (x_true * x_pred).sum(axis=1) / (np.linalg.norm(x_true, 2, 1) * np.linalg.norm(x_pred, 2, 1)+1e-5) 原本的
    #sam = (x_true * x_pred).sum(axis=1) / (np.linalg.norm(x_true, 2, 1) * np.linalg.norm(x_pred, 2, 1))
    samcos = (x_true * x_pred).sum(axis=1) / (np.linalg.norm(x_true, 2, 1) * np.linalg.norm(x_pred, 2, 1) + 1e-12)
    samcos = np.clip(samcos, -1.0, 1.0)  # 新增：裁剪
    sam = np.arccos(samcos) * 180 / np.pi
    mSAM = np.nan_to_num(sam, nan=0.0).mean()  # 新增：兜底
    
    return mSAM

def compute_psnr(x_true, x_pred):
    assert x_true.ndim == 3 and x_pred.ndim ==3

    img_w, img_h, img_c = x_true.shape
    ref = x_true.reshape(-1, img_c)
    #print(ref)
    tar = x_pred.reshape(-1, img_c)
    msr = np.mean((ref - tar) ** 2, 0) + 1e-12
    max2 = np.maximum(np.max(ref, 0) ** 2, 1e-12)
    psnrall = 10 * np.log10(max2 / msr)
    return float(np.nan_to_num(psnrall, nan=0.0, posinf=100.0).mean())


def compute_ergas(x_true, x_pred, scale_factor):
    assert x_true.ndim == 3 and x_pred.ndim ==3 and x_true.shape == x_pred.shape
    
    img_w, img_h, img_c = x_true.shape

    err = x_true - x_pred
    # err = x_pred - x_true
    ERGAS = 0
    for i in range(img_c):
        ERGAS = ERGAS + np.mean(  err[:,:,i] **2 / np.mean(x_true[:,:,i]) ** 2 + 1e-12)
    
    ERGAS = (100 / scale_factor) * np.sqrt((1/img_c) * ERGAS)
    return ERGAS


def compute_cc(x_true, x_pred):
    img_w, img_h, img_c = x_true.shape
    result = np.ones((img_c,), dtype=np.float64)
    for i in range(img_c):
        a = x_true[:, :, i].ravel().astype(np.float64)
        b = x_pred[:, :, i].ravel().astype(np.float64)
        a -= a.mean();
        b -= b.mean()
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
        result[i] = float((a @ b) / denom)
    return float(np.nan_to_num(result, nan=1.0).mean())


def compute_rmse(x_true, x_pre):
     img_w, img_h, img_c = x_true.shape
     return np.sqrt(  ((x_true-x_pre)**2).sum()/(img_w*img_h*img_c)   )

# def compute_mrae(pred, target, eps=1e-6):
#     # pred, target: [B, C, H, W] 或 [B, H, W, C]
#     abs_error = np.abs(pred - target)
#     denom = np.abs(target) + eps
#     rel_error = abs_error / denom
#     # 按空间、通道求均值（可以保留batch维或全均值）
#     return rel_error.mean()



      
def MetricsCal(x_true,x_pred, scale):# c,w,h

    sam=compute_sam(x_true, x_pred)
    
    psnr=compute_psnr(x_true, x_pred)
    
    ergas=compute_ergas(x_true, x_pred, scale)
    
    cc=compute_cc(x_true, x_pred)
    
    rmse=compute_rmse(x_true, x_pred)



    
    from skimage.metrics import structural_similarity as ssim
    ssims = []
    for i in range(x_true.shape[2]):
        ssimi = ssim(x_true[:,:,i], x_pred[:,:,i], data_range=x_pred[:,:,i].max() - x_pred[:,:,i].min())
        ssims.append(ssimi)
    Ssim = np.mean(ssims)
    
    from sewar.full_ref import uqi
    Uqi= uqi(x_true,  x_pred)
    

    return sam,psnr,ergas,cc,rmse,Ssim,Uqi
    


if __name__ == "__main__":
    pass
