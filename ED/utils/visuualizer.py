# -*- coding: utf-8 -*-            
# @Author : Hao Wei
# @Time : 2025/4/5 16:18
import math
import os
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib
import matplotlib.pyplot as plt


def plot(self, test_imgs, scores, img_scores, gt_masks, file_names, img_types, img_threshold):
    vmax = scores.max() * 255.
    vmin = scores.min() * 255. + 10
    vmax = vmax - 220
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    for i in range(len(scores)):
        img = denormalization(test_imgs[i])
        score = scores[i]
        heat_map = score * 95

        # 创建无边框画布
        fig = plt.figure(figsize=(3, 3), frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])  # 完全填充画布
        ax.set_axis_off()
        fig.add_axes(ax)

        # 绘制热力图（注意关闭插值）
        ax.imshow(heat_map, cmap='jet', norm=norm,
                  interpolation='none', aspect='auto')
        ax.imshow(img, cmap='gray', alpha=0.3,
                  interpolation='none', aspect='auto')

        # 保存配置
        save_path = self._get_save_path(img_types[i], img_scores[i], img_threshold, file_names[i])
        fig.savefig(save_path, dpi=300,
                    bbox_inches='tight',
                    pad_inches=0,
                    transparent=True)
        plt.close()


def _get_save_path(self, img_type, img_score, threshold, filename):
    if img_type == 'good':
        folder = 'normal_ok' if img_score <= threshold else 'normal_nok'
    else:
        folder = 'anomaly_ok' if img_score > threshold else 'anomaly_nok'
    return os.path.join(self.root, folder, f"{img_type}_{filename}")


def denormalization(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    mean = np.array(mean)
    std = np.array(std)
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    return x


class Visualizer(object):
    def __init__(self, root, prefix=''):
        self.root = root
        self.prefix = prefix
        os.makedirs(self.root, exist_ok=True)
        os.makedirs(os.path.join(self.root, 'normal_ok'), exist_ok=True)
        os.makedirs(os.path.join(self.root, 'normal_nok'), exist_ok=True)
        os.makedirs(os.path.join(self.root, 'anomaly_ok'), exist_ok=True)
        os.makedirs(os.path.join(self.root, 'anomaly_nok'), exist_ok=True)

    def set_prefix(self, prefix):
        self.prefix = prefix

    def plot(self, test_imgs, scores, img_scores, gt_masks, file_names, img_types, img_threshold):
        """
        Args:
            test_imgs (ndarray): shape (N, 3, h, w)
            scores (ndarray): shape (N, h, w)
            img_scores (ndarray): shape (N, )
            gt_masks (ndarray): shape (N, 1, h, w)
        """
        vmax = scores.max() * 255.
        vmin = scores.min() * 255. + 10
        vmax = vmax - 220
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        for i in range(len(scores)):
            img = test_imgs[i]
            img = denormalization(img)
            score = scores[i]
            heat_map = score * 51

            # 只创建一个子图
            fig_img, ax_img = plt.subplots(figsize=(3, 3))
            ax_img.axes.xaxis.set_visible(False)
            ax_img.axes.yaxis.set_visible(False)

            # 仅绘制热力图
            ax_img.imshow(heat_map, cmap='jet', norm=norm, interpolation='none')
            ax_img.imshow(img, cmap='gray', alpha=0, interpolation='none')

            # 保存逻辑保持不变
            if img_types[i] == 'good':
                if img_scores[i] <= img_threshold:
                    save_path = os.path.join(self.root, 'normal_ok', img_types[i] + '_' + file_names[i])
                else:
                    save_path = os.path.join(self.root, 'normal_nok', img_types[i] + '_' + file_names[i])
            else:
                if img_scores[i] > img_threshold:
                    save_path = os.path.join(self.root, 'anomaly_ok', img_types[i] + '_' + file_names[i])
                else:
                    save_path = os.path.join(self.root, 'anomaly_nok', img_types[i] + '_' + file_names[i])

            fig_img.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
            plt.close()
