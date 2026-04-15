import numpy as np
import logging
import cv2
import os
from datetime import datetime
import visdom
import numpy as np

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


class UnifiedVisualizer:
    def __init__(self, env_name="Model_Visualization", save_dir=None):
        """
        初始化统一可视化工具
        :param env_name: Visdom环境名称，同一环境的可视化会显示在同一网页
        :param save_dir: 图像保存目录
        """
        # 初始化Visdom客户端，指定环境名称
        self.vis = visdom.Visdom(env=env_name)
        if not self.vis.check_connection():
            raise ConnectionError("无法连接到Visdom服务器，请先启动Visdom服务")
        self.vis.close(env=env_name)

        self.env_name = env_name
        self.save_dir = save_dir
        if self.save_dir and not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # 初始化所有窗口
        self.loss_win = None
        self.metrics_win = {'fCP': None, 'fused': None}
        self.lr_win = None
        self.recon_win = {'fCP': None, 'fused': None}
        self.hrmsi_win = None
        self.lrhsi_win = None
        self.final_results_win = None  # 最终结果窗口

        # 记录历史数据
        self.loss_history = {
            'total': [],
            'fCP_branch': [],
            'fused_branch': []
        }
        self.metrics_history = {
            'fCP': {'psnr': [], 'sam': []},
            'fused': {'psnr': [], 'sam': []}
        }

        # 图例配置
        self.loss_legends = ['Total Loss', 'fCP Branch Loss', 'Fused Branch Loss']
        self.metric_legends = ['PSNR', 'SAM']
        self.spectral_legends = ['GT', 'CP Branch', 'Fused Branch']

        # 可视化波段配置
        # houston18:[45, 29, 13]
        # indian_pines:[9, 19, 29]
        # PaviaU:[60, 35, 9]
        # Chikusei:[4, 2, 0]、[55, 35, 15]
        # TG[29,19,6]
        self.hsi_bands = [45, 29, 13]   # houston18
        # self.hsi_bands = [58, 33, 9]
        # self.hsi_bands = [55, 35, 15]   # Chikusei
        # self.hsi_bands = [42, 86, 20]
        # self.hsi_bands = [60, 35, 9]    # PaviaU
        # self.hsi_bands = [29, 19, 6]    # TG
        self.msi_bands = [4, 2, 0]

    def update_losses(self, epoch, total_loss, fcp_loss, fused_loss):
        """更新损失可视化（总损失+两个分支损失）"""
        try:
            # 记录损失历史
            self.loss_history['total'].append(total_loss)
            self.loss_history['fCP_branch'].append(fcp_loss)
            self.loss_history['fused_branch'].append(fused_loss)

            # 准备数据
            x = np.array([epoch])
            all_losses = [total_loss, fcp_loss, fused_loss]
            max_loss = max(all_losses) * 1.2

            # 首次创建窗口
            if self.loss_win is None:
                self.loss_win = self.vis.line(
                    X=x,
                    Y=np.column_stack(all_losses),
                    opts=dict(
                        title='Stage2 Losses (Total + Branches)',
                        xlabel='Epoch',
                        ylabel='Loss Value',
                        legend=self.loss_legends,
                        ytickmin=0,
                        ytickmax=max_loss,
                        # 关键设置：确保图例可见
                        showlegend=True
                    )
                )
            else:
                # 分别更新每条曲线，每次都指定名称
                for i, (loss_val, legend) in enumerate(zip(all_losses, self.loss_legends)):
                    self.vis.line(
                        X=x,
                        Y=np.array([loss_val]),
                        win=self.loss_win,
                        name=legend,  # 严格匹配图例名称
                        update='append'
                    )

                # 动态调整y轴，同时显式保留图例配置
                all_history = (self.loss_history['total'] +
                               self.loss_history['fCP_branch'] +
                               self.loss_history['fused_branch'])
                self.vis.update_window_opts(
                    win=self.loss_win,
                    opts=dict(
                        ytickmax=max(all_history) * 1.1,
                        legend=self.loss_legends,  # 显式保留图例
                        showlegend=True
                    )
                )
            logging.info("损失可视化更新成功")
        except Exception as e:
            logging.error(f"损失可视化失败: {e}")

    def update_metrics(self, branch, epoch, psnr, sam):
        """更新指标可视化（PSNR+SAM）"""
        try:
            # 记录指标历史
            self.metrics_history[branch]['psnr'].append(psnr)
            self.metrics_history[branch]['sam'].append(sam)

            # 准备数据
            x = np.array([epoch])
            metrics = [psnr, sam]
            max_val = max(metrics) * 1.2

            if self.metrics_win[branch] is None:
                # 首次创建窗口
                self.metrics_win[branch] = self.vis.line(
                    X=x,
                    Y=np.column_stack(metrics),
                    opts=dict(
                        title=f'Stage2 Metrics ({branch} Branch)',
                        xlabel='Epoch',
                        ylabel='Value',
                        legend=self.metric_legends,
                        ytickmin=0,
                        ytickmax=max_val,
                        showlegend=True  # 确保图例可见
                    )
                )
            else:
                # 更新每条曲线
                for metric_val, legend in zip(metrics, self.metric_legends):
                    self.vis.line(
                        X=x,
                        Y=np.array([metric_val]),
                        win=self.metrics_win[branch],
                        name=legend,  # 匹配图例
                        update='append'
                    )

                # 调整y轴并保留图例
                if epoch % 100 == 0 or len(self.metrics_history[branch]['psnr']) < 5:
                    all_psnr = self.metrics_history[branch]['psnr']
                    all_sam = self.metrics_history[branch]['sam']
                    max_history = max(np.max(all_psnr) * 1.1, np.max(all_sam) * 1.1)

                    self.vis.update_window_opts(
                        win=self.metrics_win[branch],
                        opts=dict(
                            ytickmax=max_history,
                            legend=self.metric_legends,  # 显式保留图例
                            showlegend=True
                        )
                    )
            logging.info(f"{branch}分支指标可视化更新成功")
        except Exception as e:
            logging.error(f"{branch}分支指标可视化失败: {e}")

    def update_spectral_curve(self, epoch, gt_curve, fcp_curve, fused_curve):
        """更新光谱曲线对比可视化"""
        try:
            # 光谱曲线更新时也显式保留图例
            self.vis.line(
                X=np.arange(gt_curve.shape[0]),
                Y=np.stack([gt_curve, fcp_curve, fused_curve], axis=1),
                opts=dict(
                    title=f'Stage2 Spectral Curve (Epoch {epoch})',
                    xlabel='Band',
                    ylabel='Intensity',
                    legend=self.spectral_legends,
                    showlegend=True
                ),
                win='stage2_spectral'
            )

            # 强制更新窗口选项以确保图例可见
            self.vis.update_window_opts(
                win='stage2_spectral',
                opts=dict(
                    legend=self.spectral_legends,
                    showlegend=True
                )
            )
            logging.info("光谱曲线可视化更新成功")
        except Exception as e:
            logging.error(f"光谱曲线可视化失败: {e}")

    def update_learning_rate(self, epoch, lr):
        """更新学习率可视化"""
        try:
            if self.lr_win is None:
                self.lr_win = self.vis.line(
                    X=np.array([epoch]),
                    Y=np.array([lr]),
                    opts=dict(title='Stage2 Learning Rate', xlabel='Epoch', ylabel='LR')
                )
            else:
                self.vis.line(
                    X=np.array([epoch]),
                    Y=np.array([lr]),
                    win=self.lr_win,
                    update='append'
                )
            logging.info("学习率可视化更新成功")
        except Exception as e:
            logging.error(f"学习率可视化失败: {e}")

    def update_reconstructions(self, branch, epoch, img_data):
        """更新重建结果可视化"""
        try:
            # 处理图像数据
            vis_img = self._normalize(img_data[..., self.hsi_bands])
            vis_img = vis_img.transpose(2, 0, 1).astype(np.float32)

            if self.recon_win[branch] is None:
                self.recon_win[branch] = self.vis.image(
                    vis_img,
                    opts=dict(title=f'Stage2 {branch} Reconstruction', caption=f'Epoch {epoch}')
                )
            else:
                self.vis.image(
                    vis_img,
                    win=self.recon_win[branch],
                    opts=dict(title=f'Stage2 {branch} Reconstruction', caption=f'Epoch {epoch}')
                )
            logging.info(f"{branch}分支重建结果可视化更新成功")
        except Exception as e:
            logging.error(f"{branch}分支重建结果可视化失败: {e}")

    def update_hrmsi(self, epoch, fcp_data, fused_data):
        """更新HR-MSI退化结果可视化"""
        try:
            # 处理HR-MSI图像
            hr_imgs = [
                self._normalize(fcp_data[..., self.msi_bands]).transpose(2, 0, 1).astype(np.float32),
                self._normalize(fused_data[..., self.msi_bands]).transpose(2, 0, 1).astype(np.float32)
            ]
            captions = ['fCP Branch - HR-MSI', 'Fused Branch - HR-MSI']

            if self.hrmsi_win is None:
                self.hrmsi_win = self.vis.images(
                    hr_imgs,
                    nrow=2,
                    opts=dict(title=f'Stage2 HR-MSI Results (Epoch {epoch})', caption=captions)
                )
            else:
                self.vis.images(
                    hr_imgs,
                    nrow=2,
                    win=self.hrmsi_win,
                    opts=dict(title=f'Stage2 HR-MSI Results (Epoch {epoch})', caption=captions)
                )
            logging.info("HR-MSI可视化更新成功")
        except Exception as e:
            logging.error(f"HR-MSI可视化失败: {e}")

    def update_lrhsi(self, epoch, fcp_data, fused_data):
        """更新LR-HSI退化结果可视化"""
        try:
            # 处理LR-HSI图像
            lr_imgs = [
                self._normalize(fcp_data[..., self.hsi_bands]).transpose(2, 0, 1).astype(np.float32),
                self._normalize(fused_data[..., self.hsi_bands]).transpose(2, 0, 1).astype(np.float32)
            ]
            captions = ['fCP Branch - LR-HSI', 'Fused Branch - LR-HSI']

            if self.lrhsi_win is None:
                self.lrhsi_win = self.vis.images(
                    lr_imgs,
                    nrow=2,
                    opts=dict(title=f'Stage2 LR-HSI Results (Epoch {epoch})', caption=captions)
                )
            else:
                self.vis.images(
                    lr_imgs,
                    nrow=2,
                    win=self.lrhsi_win,
                    opts=dict(title=f'Stage2 LR-HSI Results (Epoch {epoch})', caption=captions)
                )
            logging.info("LR-HSI可视化更新成功")
        except Exception as e:
            logging.error(f"LR-HSI可视化失败: {e}")

    def update_spectral_curve(self, epoch, gt_curve, fcp_curve, fused_curve):
        """更新光谱曲线对比可视化"""
        try:
            self.vis.line(
                X=np.arange(gt_curve.shape[0]),
                Y=np.stack([gt_curve, fcp_curve, fused_curve], axis=1),
                opts=dict(
                    title=f'Stage2 Spectral Curve (Epoch {epoch})',
                    xlabel='Band',
                    ylabel='Intensity',
                    legend=['GT', 'CP Branch', 'Fused Branch']
                ),
                win='stage2_spectral'
            )
            logging.info("光谱曲线可视化更新成功")
        except Exception as e:
            logging.error(f"光谱曲线可视化失败: {e}")

    def _normalize(self, img):
        """图像标准化到0-1范围"""
        min_val = img.min()
        max_val = img.max()
        if max_val - min_val < 1e-8:
            logging.warning("图像所有像素值相同，无法正常标准化")
            return np.zeros_like(img)
        return (img - min_val) / (max_val - min_val)

    def _calculate_mrae(self, pred, gt):
        """计算相对绝对误差均值(MRAE)热图，增强数值稳定性"""
        eps = 1e-8
        # 计算绝对误差
        abs_error = np.abs(pred - gt)
        # 处理分母接近0的情况（避免除以极小值导致数值爆炸）
        gt_clamped = np.maximum(gt, eps)  # 确保分母不小于eps
        # 计算相对误差并在通道维度求均值
        mrae_map = np.mean(abs_error / gt_clamped, axis=-1)
        # 截断异常值（避免极端值主导热图）
        max_mrae = np.percentile(mrae_map, 99)  # 取99分位值作为上限
        return mrae_map

    def _calculate_sam(self, pred, gt):
        """计算光谱角映射(SAM)热图"""
        eps = 1e-8
        sam_map = np.zeros((pred.shape[0], pred.shape[1]))

        for i in range(pred.shape[0]):
            for j in range(pred.shape[1]):
                pred_vec = pred[i, j]
                gt_vec = gt[i, j]

                dot_product = np.dot(pred_vec, gt_vec)
                norm_pred = np.linalg.norm(pred_vec)
                norm_gt = np.linalg.norm(gt_vec)

                cos_theta = np.clip(dot_product / (norm_pred * norm_gt + eps), -1, 1)
                sam_map[i, j] = np.arccos(cos_theta)

        return sam_map

    def _calculate_residual(self, pred, gt):
        """计算残差热图 (预测 - 真实)"""
        abs_error = np.abs(pred - gt)
        return np.mean(abs_error, axis=-1)

    def _apply_jet_cmap(self, heatmap):
        """将单通道热图转换为jet色系的3通道图像（用于Visdom显示）"""
        import matplotlib.cm as cm
        jet_cmap = cm.get_cmap('jet')
        # 应用颜色映射并提取RGB通道（去掉alpha通道）
        heatmap_rgb = jet_cmap(heatmap)[:, :, :3]
        # 转为CHW格式（Visdom要求）
        return heatmap_rgb.transpose(2, 0, 1).astype(np.float32)

    def visualize_final_results(self, pred_img, gt_img, img_name="final_result",bands=[43, 87, 21], title_suffix=""):
        """
        在同一Visdom环境中可视化最终结果
        houston18:[45, 29, 13]
        indian_pines:[9, 19, 29]
        PaviaU:[60, 35, 9]

        """
        try:
            if pred_img.shape != gt_img.shape:
                raise ValueError(f"预测图像与GT形状不匹配: {pred_img.shape} vs {gt_img.shape}")

            # 提取可视化波段并标准化
            pred_vis = self._normalize(pred_img[..., bands]).transpose(2, 0, 1)  # 转为CHW格式
            gt_vis = self._normalize(gt_img[..., bands]).transpose(2, 0, 1)

            # 计算评估热图
            residual_map = self._calculate_residual(pred_img, gt_img)
            mrae_map = self._calculate_mrae(pred_img, gt_img)
            sam_map = self._calculate_sam(pred_img, gt_img)
            sam_map_deg = sam_map * 180 / np.pi  # SAM转成角度，便于物理量展示

            # 对每个热图单独归一化（关键修改）
            residual_norm = self._normalize(residual_map)
            mrae_norm = self._normalize(mrae_map)  # 独立归一化MRAE
            sam_norm = self._normalize(sam_map)

            # 将单通道热图转为3通道（使用jet色系，蓝→绿→黄→红）
            residual_vis = self._apply_jet_cmap(residual_norm)
            mrae_vis = self._apply_jet_cmap(mrae_norm)  # 应用jet色系
            sam_vis = self._apply_jet_cmap(sam_norm)

            # RMSE热图
            rmse_map = np.sqrt(np.mean((pred_img - gt_img) ** 2, axis=-1))
            rmse_norm = self._normalize(rmse_map)
            rmse_rgb = self._apply_jet_cmap(rmse_norm)  # 应用jet色系
            rmse_vis = self._apply_jet_cmap(rmse_norm)


            # 准备所有要显示的图像
            images = [
                pred_vis,  # 预测图像
                gt_vis,  # GT图像
                rmse_vis,  # RMSE热图
                residual_vis,  # 残差热图
                mrae_vis,  # MRAE热图
                sam_vis  # SAM热图
            ]

            # 图像标题
            captions = [
                "Pred-Image ",
                "GT ",
                "RMSE Heatmap ",
                "Residual Heatmap ",
                "MRAE Heatmap ",
                "SAM Heatmap "
            ]

            # 在Visdom中显示（3行2列布局）
            if self.final_results_win is None:
                self.final_results_win = self.vis.images(
                    images,
                    nrow=3,  # 每行显示3张图
                    opts=dict(
                        title=f"Final Results {title_suffix}",
                        caption=captions,
                        showlegend=True,
                    )
                )
            else:
                self.vis.images(
                    images,
                    nrow=3,
                    win=self.final_results_win,
                    opts=dict(
                        title=f"Final Results {title_suffix}",
                        caption=captions
                    )
                )


            if self.save_dir:
                import matplotlib.pyplot as plt

                fig, axes = plt.subplots(5, 2, figsize=(12, 18))
                fig.suptitle(f"Pred, GT and Error Maps: Physical Value (Left) vs Normalized (Right) {title_suffix}",
                                 fontsize=15)

                # 预测和GT图（归一化RGB，HWC）
                pred_rgb = pred_vis.transpose(1, 2, 0)
                gt_rgb = gt_vis.transpose(1, 2, 0)

                # RMSE/Residual/MRAE/SAM物理量与归一化
                rmse_map = np.sqrt(np.mean((pred_img - gt_img) ** 2, axis=-1))
                residual_map = self._calculate_residual(pred_img, gt_img)
                mrae_map = self._calculate_mrae(pred_img, gt_img)
                sam_map = self._calculate_sam(pred_img, gt_img)
                sam_map_deg = sam_map * 180 / np.pi

                def norm_map_func(arr):
                    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-8)

                rmse_norm = norm_map_func(rmse_map)
                residual_norm = norm_map_func(residual_map)
                mrae_norm = norm_map_func(mrae_map)
                sam_norm = norm_map_func(sam_map)

                # 准备行内容
                data_rows = [
                    # (左, 右, 左标题, 右标题, 是否热力图)
                    (pred_rgb, gt_rgb, 'Pred-Image', 'GT', False),
                    (rmse_map, rmse_norm, 'RMSE (Phys)', 'RMSE (Norm)', True),
                    (residual_map, residual_norm, 'Residual (Phys)', 'Residual (Norm)', True),
                    (mrae_map, mrae_norm, 'MRAE (Phys)', 'MRAE (Norm)', True),
                    (sam_map_deg, sam_norm, 'SAM (Deg)', 'SAM (Norm)', True),
                ]

                def get_vmax(arr):
                    return np.percentile(arr, 99)

                for row, (left_img, right_img, left_title, right_title, is_heatmap) in enumerate(data_rows):
                    # 左
                    if is_heatmap:
                        vmax = get_vmax(left_img)
                        im_left = axes[row, 0].imshow(left_img, cmap='jet', vmin=0, vmax=vmax)
                        cbar_left = fig.colorbar(im_left, ax=axes[row, 0], fraction=0.046, pad=0.04)
                        min_val, max_val = np.min(left_img), np.max(left_img)
                        cbar_left.set_label(f'{left_title}\n[min={min_val:.3f}, max={max_val:.3f}]', rotation=270,
                                            labelpad=25, fontsize=10)
                    else:
                        axes[row, 0].imshow(left_img)
                    axes[row, 0].set_title(left_title)
                    axes[row, 0].axis('off')

                    # 右
                    if is_heatmap:
                        im_right = axes[row, 1].imshow(right_img, cmap='jet', vmin=0, vmax=1)
                        cbar_right = fig.colorbar(im_right, ax=axes[row, 1], fraction=0.046, pad=0.04)
                        cbar_right.set_label('Normalized', rotation=270, labelpad=15)
                    else:
                        axes[row, 1].imshow(right_img)
                    axes[row, 1].set_title(right_title)
                    axes[row, 1].axis('off')

                plt.tight_layout(rect=[0, 0.03, 1, 0.97])
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(self.save_dir, f"{img_name}_results_dualcol_{timestamp}.png")
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                logging.info(f"结果图像已保存至: {save_path}")

            return {
                'mean_residual': np.mean(residual_map),
                'mean_mrae': np.mean(mrae_map),
                'mean_sam': np.mean(sam_map),
                'max_sam': np.max(sam_map)
            }

        except Exception as e:
            logging.error(f"最终结果可视化失败: {e}")
            return None
