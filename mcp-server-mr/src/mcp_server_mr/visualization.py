# src/mcp_server_mr/visualization.py
"""
Visualization module for MR analysis results.
Generates plots and converts them to base64 for transmission.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
import pandas as pd
import base64
import io
from typing import List, Optional
import logging

from .models import HarmonizedDataPoint, MRMethodResult

logger = logging.getLogger(__name__)

# Set matplotlib to use a non-interactive backend
plt.switch_backend('Agg')

class MRVisualizer:
    """
    Class for generating MR analysis visualizations.
    """
    
    def __init__(self, data: List[HarmonizedDataPoint]):
        """
        Initialize visualizer with harmonized data.
        
        Args:
            data: List of harmonized SNP data points
        """
        self.data = data
        self.df = self._prepare_dataframe()
        
    def _prepare_dataframe(self) -> pd.DataFrame:
        """Convert harmonized data to pandas DataFrame."""
        data_dict = {
            'SNP': [d.SNP for d in self.data],
            'beta_exposure': [d.beta_exposure for d in self.data],
            'se_exposure': [d.se_exposure for d in self.data],
            'beta_outcome': [d.beta_outcome for d in self.data],
            'se_outcome': [d.se_outcome for d in self.data],
        }
        return pd.DataFrame(data_dict)
    
    def _fig_to_base64(self, fig, format='png') -> str:
        """Convert matplotlib figure to base64 string."""
        buffer = io.BytesIO()
        if format == 'svg':
            fig.savefig(buffer, format='svg', bbox_inches='tight')
        else:
            fig.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        buffer.close()
        plt.close(fig)
        return image_base64

    def _fig_to_file(self, fig, filepath, format='png'):
        """Save matplotlib figure to file."""
        if format == 'svg':
            fig.savefig(filepath, format='svg', bbox_inches='tight')
        else:
            fig.savefig(filepath, format='png', dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    def _format_p_value(self, p_value: float) -> str:
        """格式化P值 - 统一处理极小P值和正常P值"""
        # 处理特殊情况
        if p_value == 0.0 or p_value < 1e-100:
            return "0.0e+00"
        elif p_value < 1e-10:
            return f"{p_value:.1e}"
        elif p_value < 0.001:
            return f"{p_value:.1e}"
        else:
            return f"{p_value:.3f}"

    def _find_mr_result_by_method(self, mr_results: List[MRMethodResult], method_keywords: List[str]) -> Optional[MRMethodResult]:
        """根据方法关键词查找MR结果，确保数据正确性"""
        for result in mr_results:
            method_lower = result.method.lower()
            # 特殊处理：确保精确匹配，避免"weighted"关键词匹配到IVW
            if method_keywords == ['weighted', 'median']:
                # 只匹配包含"median"的方法
                if 'median' in method_lower and 'weighted' in method_lower:
                    return result
            elif method_keywords == ['inverse', 'variance', 'weighted', 'ivw']:
                # 只匹配IVW相关的方法，排除median
                if ('inverse' in method_lower or 'ivw' in method_lower) and 'median' not in method_lower:
                    return result
            else:
                # 其他方法使用原来的逻辑
                if any(keyword.lower() in method_lower for keyword in method_keywords):
                    return result
        return None

    def create_scatter_plot(self, mr_results: List[MRMethodResult]) -> Optional[str]:
        """
        Create scatter plot showing SNP effects on exposure vs outcome.
        Style matches 生成案例1最新图表.py exactly.

        Args:
            mr_results: Results from MR analysis

        Returns:
            Base64-encoded scatter plot image
        """
        try:
            logger.info("Creating scatter plot")

            # 使用与案例1完全相同的尺寸设置
            fig, ax = plt.subplots(figsize=(7, 5.5), dpi=300)

            # 提取数据
            x_data = self.df['beta_exposure'].values
            y_data = self.df['beta_outcome'].values

            # 统一样式 - 所有SNP都是基因组显著的
            colors = ['#1f77b4'] * len(x_data)
            sizes = [60] * len(x_data)

            # 绘制散点（不显示标签）
            scatter = ax.scatter(x_data, y_data, c=colors, s=sizes, alpha=0.7,
                               edgecolors='none', linewidth=0, zorder=3)

            # 添加MR方法回归线 - 根据方法名称匹配真实数据
            x_range = np.linspace(min(x_data) * 1.3, max(x_data) * 1.3, 100)

            # 绘制IVW回归线
            ivw_result = self._find_mr_result_by_method(mr_results, ['inverse', 'variance', 'weighted', 'ivw'])
            if ivw_result:
                y_line = ivw_result.estimate * x_range
                ax.plot(x_range, y_line, color='#d62728', linestyle='-', linewidth=2.5, alpha=0.8, zorder=2)

            # 绘制MR-Egger回归线
            egger_result = self._find_mr_result_by_method(mr_results, ['egger'])
            if egger_result:
                y_line = egger_result.estimate * x_range
                ax.plot(x_range, y_line, color='#2ca02c', linestyle='--', linewidth=2.5, alpha=0.8, zorder=2)

            # 绘制Weighted Median回归线
            median_result = self._find_mr_result_by_method(mr_results, ['weighted', 'median'])
            if median_result:
                y_line = median_result.estimate * x_range
                ax.plot(x_range, y_line, color='#ff7f0e', linestyle='-.', linewidth=2.5, alpha=0.8, zorder=2)

            # 使用与案例一相同的标签和标题样式
            ax.set_xlabel('SNP effect on exposure (β)', fontweight='bold')
            ax.set_ylabel('SNP effect on outcome (β)', fontweight='bold')
            ax.set_title('Mendelian Randomization Scatter Plot', fontweight='bold', pad=15)

            # 创建图例 - 根据方法名称匹配真实数据，确保数据正确性
            legend_elements = []

            # 查找IVW结果
            ivw_result = self._find_mr_result_by_method(mr_results, ['inverse', 'variance', 'weighted', 'ivw'])
            if ivw_result:
                # 添加调试信息
                logger.info(f"散点图 - IVW: β={ivw_result.estimate:.4f}, P={ivw_result.p_value:.6e}")
                legend_elements.append(
                    plt.Line2D([0], [0], color='#d62728', linestyle='-', linewidth=2.5,
                               label=f"Inverse variance weighted: β = {ivw_result.estimate:.4f} (P = {self._format_p_value(ivw_result.p_value)})")
                )

            # 查找MR-Egger结果
            egger_result = self._find_mr_result_by_method(mr_results, ['egger'])
            if egger_result:
                # 添加调试信息
                logger.info(f"散点图 - MR-Egger: β={egger_result.estimate:.4f}, P={egger_result.p_value:.6e}")
                legend_elements.append(
                    plt.Line2D([0], [0], color='#2ca02c', linestyle='--', linewidth=2.5,
                               label=f"MR Egger: β = {egger_result.estimate:.4f} (P = {self._format_p_value(egger_result.p_value)})")
                )

            # 查找Weighted Median结果
            median_result = self._find_mr_result_by_method(mr_results, ['weighted', 'median'])
            if median_result:
                # 添加调试信息
                logger.info(f"散点图 - Weighted Median: β={median_result.estimate:.4f}, P={median_result.p_value:.6e}")
                legend_elements.append(
                    plt.Line2D([0], [0], color='#ff7f0e', linestyle='-.', linewidth=2.5,
                               label=f"Weighted median: β = {median_result.estimate:.4f} (P = {self._format_p_value(median_result.p_value)})")
                )

            # 添加SNP信息
            legend_elements.append(
                plt.scatter([], [], c='#1f77b4', s=60,
                           label=f'SNPs (n={len(x_data)})')
            )

            legend = ax.legend(handles=legend_elements, loc='upper left', frameon=True, fancybox=False, shadow=False,
                              framealpha=0.9)
            legend.get_frame().set_facecolor('white')
            legend.get_frame().set_edgecolor('black')
            legend.get_frame().set_linewidth(0.5)

            # 网格和样式优化 - 与案例1完全相同
            ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            plt.tight_layout()

            return self._fig_to_base64(fig)

        except Exception as e:
            logger.error(f"Error creating scatter plot: {e}")
            return None

    def _add_jitter_to_points(self, x_data, y_data, jitter_strength=0.001):
        """
        为重叠的点添加微小抖动，提高可视性
        """
        try:
            # 检测重叠点
            x_jittered = x_data.copy()
            y_jittered = y_data.copy()

            # 计算数据范围用于确定抖动强度
            x_range = np.max(x_data) - np.min(x_data)
            y_range = np.max(y_data) - np.min(y_data)

            # 自适应抖动强度
            x_jitter_amount = x_range * jitter_strength
            y_jitter_amount = y_range * jitter_strength

            # 为每个点添加随机抖动
            np.random.seed(42)  # 确保结果可重现
            x_jittered += np.random.normal(0, x_jitter_amount, len(x_data))
            y_jittered += np.random.normal(0, y_jitter_amount, len(y_data))

            return x_jittered, y_jittered

        except Exception as e:
            logger.warning(f"Error adding jitter: {e}")
            return x_data, y_data

    def _add_smart_labels(self, ax, df, x_coords=None, y_coords=None):
        """
        智能添加SNP标签，使用高级防重叠算法
        """
        try:
            # 使用抖动后的坐标（如果提供）
            if x_coords is None:
                x_coords = df['beta_exposure'].values
            if y_coords is None:
                y_coords = df['beta_outcome'].values

            n_points = len(df)

            if n_points <= 3:
                # 3个点以下，简单放置
                self._place_simple_labels(ax, df, x_coords, y_coords)
            elif n_points <= 8:
                # 4-8个点，使用防重叠算法
                self._place_non_overlapping_labels(ax, df, x_coords, y_coords)
            else:
                # 超过8个点，只显示关键点
                self._place_key_labels_only(ax, df, x_coords, y_coords)

        except Exception as e:
            logger.warning(f"Error adding smart labels: {e}")
            # 降级到最简单的标签
            self._place_fallback_labels(ax, df, x_coords, y_coords)

    def _place_simple_labels(self, ax, df, x_coords, y_coords):
        """为少量点放置简单标签"""
        for i, row in df.iterrows():
            ax.annotate(row['SNP'],
                       (x_coords[i], y_coords[i]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=7, alpha=0.9, fontweight='medium',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                alpha=0.9, edgecolor='gray', linewidth=0.5))

    def _place_non_overlapping_labels(self, ax, df, x_coords, y_coords):
        """使用防重叠算法放置标签 - 只为实际的数据点放置标签"""

        # 首先检查实际有多少个唯一的点位置
        unique_positions = []
        position_to_snps = {}

        # 将相近的点归为一组（容差范围内）
        tolerance = 0.001  # 调整容差

        for i, row in df.iterrows():
            x, y = x_coords[i], y_coords[i]

            # 检查是否与已有位置重叠
            found_group = False
            for pos_key, snp_list in position_to_snps.items():
                pos_x, pos_y = pos_key
                if abs(x - pos_x) < tolerance and abs(y - pos_y) < tolerance:
                    # 归入现有组
                    snp_list.append((i, row['SNP']))
                    found_group = True
                    break

            if not found_group:
                # 创建新组
                pos_key = (x, y)
                position_to_snps[pos_key] = [(i, row['SNP'])]
                unique_positions.append(pos_key)

        logger.debug(f"发现 {len(unique_positions)} 个唯一位置，总共 {len(df)} 个SNP")

        # 为每个唯一位置放置标签
        placed_labels = []

        for pos_idx, (pos_x, pos_y) in enumerate(unique_positions):
            snp_group = position_to_snps[(pos_x, pos_y)]

            if len(snp_group) == 1:
                # 单个SNP，简单放置
                i, snp_name = snp_group[0]
                offset = (0, 12) if pos_idx % 2 == 0 else (0, -12)

                ax.annotate(snp_name,
                           (pos_x, pos_y),
                           xytext=offset, textcoords='offset points',
                           fontsize=7, alpha=0.9,
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='lightblue',
                                    alpha=0.8, edgecolor='blue', linewidth=0.5))
            else:
                # 多个SNP在同一位置，需要分散显示
                logger.debug(f"位置 ({pos_x:.4f}, {pos_y:.4f}) 有 {len(snp_group)} 个SNP")

                # 为同一位置的多个SNP创建上下错开的标签
                for snp_idx, (i, snp_name) in enumerate(snp_group):
                    # 上下交替放置
                    if snp_idx == 0:
                        offset = (0, 15)
                    elif snp_idx == 1:
                        offset = (0, -15)
                    elif snp_idx == 2:
                        offset = (8, 20)
                    elif snp_idx == 3:
                        offset = (-8, -20)
                    else:
                        # 更多SNP时使用圆形分布
                        angle = snp_idx * 2 * np.pi / len(snp_group)
                        offset = (15 * np.cos(angle), 15 * np.sin(angle))

                    ax.annotate(snp_name,
                               (pos_x, pos_y),
                               xytext=offset, textcoords='offset points',
                               fontsize=6, alpha=0.85,
                               bbox=dict(boxstyle='round,pad=0.15', facecolor='lightyellow',
                                        alpha=0.8, edgecolor='orange', linewidth=0.4))

                    placed_labels.append({
                        'x': pos_x,
                        'y': pos_y,
                        'offset': offset,
                        'text': snp_name
                    })
    def _calculate_label_overlap(self, x, y, offset, placed_labels):
        """计算标签重叠度分数"""
        if not placed_labels:
            return 0

        # 当前标签的预期位置
        current_label_x = x + offset[0] / 100  # 转换为数据坐标
        current_label_y = y + offset[1] / 100

        overlap_score = 0
        for placed in placed_labels:
            # 已放置标签的位置
            placed_x = placed['x'] + placed['offset'][0] / 100
            placed_y = placed['y'] + placed['offset'][1] / 100

            # 计算距离
            distance = np.sqrt((current_label_x - placed_x)**2 + (current_label_y - placed_y)**2)

            # 距离越近，重叠分数越高
            if distance < 0.02:  # 阈值可调整
                overlap_score += 1 / (distance + 0.001)

        return overlap_score

    def _place_key_labels_only(self, ax, df, x_coords, y_coords):
        """只为关键点放置标签"""
        # 找到最极端的点
        extreme_indices = []

        # 最大和最小的exposure效应
        extreme_indices.extend([df['beta_exposure'].idxmax(), df['beta_exposure'].idxmin()])
        # 最大和最小的outcome效应
        extreme_indices.extend([df['beta_outcome'].idxmax(), df['beta_outcome'].idxmin()])

        # 去重并限制数量
        extreme_indices = list(set(extreme_indices))[:3]

        # 为极值点使用不重叠的偏移
        key_offsets = [(10, 10), (-10, 10), (10, -10)]

        for idx, i in enumerate(extreme_indices):
            if i < len(df):
                row = df.iloc[i]
                offset = key_offsets[idx % len(key_offsets)]
                ax.annotate(row['SNP'],
                           (x_coords[i], y_coords[i]),
                           xytext=offset, textcoords='offset points',
                           fontsize=6, alpha=0.9, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.15', facecolor='red',
                                    alpha=0.7, edgecolor='darkred', linewidth=0.5))

        # 添加说明
        ax.text(0.02, 0.98, f'{len(extreme_indices)}/{len(df)} key SNPs labeled',
               transform=ax.transAxes, fontsize=6, alpha=0.8,
               verticalalignment='top',
               bbox=dict(boxstyle='round,pad=0.2', facecolor='lightgray', alpha=0.6))

    def _place_fallback_labels(self, ax, df, x_coords, y_coords):
        """降级标签放置（最简单的方式）"""
        # 只显示前3个点的标签
        for i in range(min(3, len(df))):
            row = df.iloc[i]
            ax.annotate(row['SNP'],
                       (x_coords[i], y_coords[i]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=6, alpha=0.7,
                       bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.8))

    def _add_smart_labels_for_funnel(self, ax, x_coords, y_coords, snp_names):
        """为漏斗图添加增强的防重叠标签 - 确保所有点都有标签"""
        try:
            n_points = len(snp_names)
            logger.debug(f"漏斗图有 {n_points} 个SNP需要标注")

            if n_points <= 3:
                # 少量点，所有点都显示标签
                self._place_funnel_all_labels_small(ax, x_coords, y_coords, snp_names)

            elif n_points <= 8:
                # 中等数量点，所有点都显示标签，使用增强防重叠
                self._place_funnel_all_labels_medium(ax, x_coords, y_coords, snp_names)

            else:
                # 大量点，显示所有点但使用紧凑标签
                self._place_funnel_all_labels_compact(ax, x_coords, y_coords, snp_names)

        except Exception as e:
            logger.warning(f"Error adding funnel plot smart labels: {e}")
            # 确保降级时也显示所有标签
            self._place_funnel_all_labels_fallback(ax, x_coords, y_coords, snp_names)

    def _place_funnel_labels_enhanced(self, ax, x_coords, y_coords, snp_names):
        """增强的漏斗图标签放置算法"""
        # 计算点之间的距离，避免重叠
        placed_labels = []

        for i, snp in enumerate(snp_names):
            x, y = x_coords[i], y_coords[i]

            # 候选偏移位置 - 专门为漏斗图优化
            candidate_offsets = [
                (0, 20), (0, -20),                    # 上下优先，大间距
                (15, 15), (-15, 15), (15, -15), (-15, -15),  # 对角线，大间距
                (20, 0), (-20, 0),                    # 左右，大间距
                (8, 25), (-8, 25), (8, -25), (-8, -25),      # 远距离上下
                (25, 8), (-25, 8), (25, -8), (-25, -8)       # 远距离左右
            ]

            best_offset = candidate_offsets[0]
            min_overlap = float('inf')

            # 为每个候选位置计算重叠度
            for offset in candidate_offsets:
                overlap_score = self._calculate_funnel_label_overlap(x, y, offset, placed_labels)
                if overlap_score < min_overlap:
                    min_overlap = overlap_score
                    best_offset = offset

            # 放置标签
            ax.annotate(snp, (x, y),
                       xytext=best_offset, textcoords='offset points',
                       fontsize=6, alpha=0.85,
                       bbox=dict(boxstyle='round,pad=0.15', facecolor='lightyellow',
                                alpha=0.8, edgecolor='orange', linewidth=0.4))

            # 记录已放置的标签
            placed_labels.append({'x': x, 'y': y, 'offset': best_offset, 'text': snp})

    def _calculate_funnel_label_overlap(self, x, y, offset, placed_labels):
        """计算漏斗图标签重叠度"""
        if not placed_labels:
            return 0

        # 当前标签的预期位置（转换为数据坐标的近似值）
        current_x = x + offset[0] * 0.01  # 粗略转换
        current_y = y + offset[1] * 0.01

        overlap_score = 0
        for placed in placed_labels:
            placed_x = placed['x'] + placed['offset'][0] * 0.01
            placed_y = placed['y'] + placed['offset'][1] * 0.01

            # 计算距离
            distance = np.sqrt((current_x - placed_x)**2 + (current_y - placed_y)**2)

            # 距离越近，重叠分数越高
            if distance < 0.05:  # 漏斗图的阈值
                overlap_score += 1 / (distance + 0.001)

        return overlap_score

    def _place_funnel_key_labels(self, ax, x_coords, y_coords, snp_names):
        """只为漏斗图的关键点放置标签"""
        # 找到最极端的点
        extreme_indices = []

        # 最左和最右的点
        extreme_indices.extend([np.argmin(x_coords), np.argmax(x_coords)])
        # 最高精度的点
        extreme_indices.append(np.argmax(y_coords))

        # 去重并限制数量
        extreme_indices = list(set(extreme_indices))[:3]

        key_offsets = [(0, 20), (0, -20), (15, 15)]

        for idx, i in enumerate(extreme_indices):
            snp = snp_names[i]
            offset = key_offsets[idx % len(key_offsets)]
            ax.annotate(snp, (x_coords[i], y_coords[i]),
                       xytext=offset, textcoords='offset points',
                       fontsize=6, alpha=0.9, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.15', facecolor='red',
                                alpha=0.7, edgecolor='darkred', linewidth=0.5))

    def _place_funnel_all_labels_small(self, ax, x_coords, y_coords, snp_names):
        """少量点 - 所有点都显示标签，适中距离"""
        offsets = [(0, 12), (0, -12), (10, 8), (-10, -8), (12, 0), (-12, 0)]
        for i, snp in enumerate(snp_names):
            offset = offsets[i % len(offsets)]
            ax.annotate(snp, (x_coords[i], y_coords[i]),
                       xytext=offset, textcoords='offset points',
                       fontsize=7, alpha=0.9,
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='lightblue',
                                alpha=0.9, edgecolor='blue', linewidth=0.5))

    def _place_funnel_all_labels_medium(self, ax, x_coords, y_coords, snp_names):
        """中等数量点 - 所有点都显示标签，防重叠算法"""
        placed_labels = []

        for i, snp in enumerate(snp_names):
            x, y = x_coords[i], y_coords[i]

            # 优化的候选偏移位置 - 减少距离，保持清晰
            candidate_offsets = [
                (0, 15), (0, -15),                    # 上下，适中距离
                (12, 12), (-12, 12), (12, -12), (-12, -12),  # 对角线，适中距离
                (16, 0), (-16, 0),                    # 左右，适中距离
                (8, 18), (-8, 18), (8, -18), (-8, -18),      # 混合位置
                (18, 8), (-18, 8), (18, -8), (-18, -8),      # 更多混合位置
                (10, 20), (-10, 20), (10, -20), (-10, -20),  # 稍远位置
                (20, 10), (-20, 10), (20, -10), (-20, -10)   # 备选位置
            ]

            best_offset = candidate_offsets[0]
            min_overlap = float('inf')

            # 为每个候选位置计算重叠度
            for offset in candidate_offsets:
                overlap_score = self._calculate_funnel_label_overlap(x, y, offset, placed_labels)
                if overlap_score < min_overlap:
                    min_overlap = overlap_score
                    best_offset = offset

            # 放置标签
            ax.annotate(snp, (x, y),
                       xytext=best_offset, textcoords='offset points',
                       fontsize=6, alpha=0.85,
                       bbox=dict(boxstyle='round,pad=0.15', facecolor='lightyellow',
                                alpha=0.8, edgecolor='orange', linewidth=0.4))

            # 记录已放置的标签
            placed_labels.append({'x': x, 'y': y, 'offset': best_offset, 'text': snp})

    def _place_funnel_all_labels_compact(self, ax, x_coords, y_coords, snp_names):
        """大量点 - 所有点都显示标签，紧凑模式"""
        # 使用圆形分布来避免重叠
        import math

        for i, snp in enumerate(snp_names):
            # 计算圆形分布的角度
            angle = (i * 2 * math.pi) / len(snp_names)
            radius = 14  # 减小半径，拉近标签距离

            # 根据角度计算偏移 - 使用较小的半径
            offset_x = radius * math.cos(angle)
            offset_y = radius * math.sin(angle)

            ax.annotate(snp, (x_coords[i], y_coords[i]),
                       xytext=(offset_x, offset_y), textcoords='offset points',
                       fontsize=5, alpha=0.8,
                       bbox=dict(boxstyle='round,pad=0.1', facecolor='lightgray',
                                alpha=0.7, edgecolor='gray', linewidth=0.3))

    def _place_funnel_all_labels_fallback(self, ax, x_coords, y_coords, snp_names):
        """降级模式 - 确保所有点都有标签"""
        # 简单的交替模式，适中距离
        offsets = [
            (0, 12), (0, -12), (12, 0), (-12, 0),
            (10, 10), (-10, 10), (10, -10), (-10, -10),
            (14, 0), (-14, 0), (0, 14), (0, -14)
        ]

        for i, snp in enumerate(snp_names):
            offset = offsets[i % len(offsets)]
            ax.annotate(snp, (x_coords[i], y_coords[i]),
                       xytext=offset, textcoords='offset points',
                       fontsize=6, alpha=0.7,
                       bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.8))
    
    def create_forest_plot(self, mr_results: List[MRMethodResult]) -> Optional[str]:
        """
        Create forest plot showing results from different MR methods.
        Style matches 生成案例1最新图表.py exactly.

        Args:
            mr_results: Results from MR analysis

        Returns:
            Base64-encoded forest plot image
        """
        try:
            logger.info("Creating forest plot")

            if not mr_results:
                return None

            # 使用与案例1完全相同的森林图尺寸
            fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

            # 数据准备
            n_studies = len(mr_results)
            y_positions = np.arange(n_studies)

            # 颜色设置 - 与案例1完全相同
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

            # 绘制每个研究的结果 - 与案例1完全相同的样式
            for i, (result, color) in enumerate(zip(mr_results, colors)):
                y = n_studies - 1 - i  # 从上到下

                # 绘制置信区间线
                ax.plot([result.ci_lower, result.ci_upper], [y, y],
                       color=color, linewidth=3, alpha=0.7, zorder=2)

                # 绘制置信区间端点
                ax.plot([result.ci_lower, result.ci_lower], [y-0.1, y+0.1],
                       color=color, linewidth=3, alpha=0.7, zorder=2)
                ax.plot([result.ci_upper, result.ci_upper], [y-0.1, y+0.1],
                       color=color, linewidth=3, alpha=0.7, zorder=2)

                # 绘制点估计
                ax.plot(result.estimate, y, 's', markersize=8, color=color,
                       markeredgecolor='white', markeredgewidth=1.5, zorder=3)

                # 在线的上方添加统计信息
                stats_text = f"β = {result.estimate:.4f} [{result.ci_lower:.4f}, {result.ci_upper:.4f}]  P = {self._format_p_value(result.p_value)}"

                ax.text(result.estimate, y + 0.35, stats_text,
                       fontsize=9, va='bottom', ha='center',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8,
                                edgecolor=color, linewidth=1))

            # 添加零效应线
            ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5, zorder=1)

            # 设置y轴标签 - 与案例1完全相同
            method_labels = [result.method for result in reversed(mr_results)]
            ax.set_yticks(y_positions)
            ax.set_yticklabels(method_labels, fontsize=11, fontweight='bold')

            # 设置x轴 - 与案例1完全相同
            ax.set_xlabel('Causal Effect Estimate (β) with 95% CI', fontweight='bold', fontsize=12)

            # 设置标题 - 与案例1完全相同
            ax.set_title('Mendelian Randomization Forest Plot',
                        fontweight='bold', fontsize=14, pad=20)

            # 设置坐标轴范围
            all_ci_lower = [r.ci_lower for r in mr_results]
            all_ci_upper = [r.ci_upper for r in mr_results]
            x_min = min(all_ci_lower) - 0.001
            x_max = max(all_ci_upper) + 0.001
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(-0.5, n_studies - 0.3)

            # 美化网格
            ax.grid(True, axis='x', alpha=0.3, linestyle=':', linewidth=0.5)
            ax.set_axisbelow(True)

            plt.tight_layout()
            
            return self._fig_to_base64(fig)
            
        except Exception as e:
            logger.error(f"Error creating forest plot: {e}")
            return None
    
    def create_funnel_plot(self, mr_results: List[MRMethodResult] = None) -> Optional[str]:
        """
        Create funnel plot for assessing publication bias.
        Style matches 生成案例1最新图表.py exactly.

        Returns:
            Base64-encoded funnel plot image
        """
        try:
            logger.info("Creating funnel plot")

            # 使用与案例1完全相同的漏斗图尺寸
            fig, ax = plt.subplots(figsize=(7, 6), dpi=300)

            # 计算效应大小和精度 - 与案例1完全相同的方法
            effect_sizes = []
            precisions = []
            colors = []

            for i in range(len(self.df)):
                if self.df.iloc[i]['beta_exposure'] != 0:
                    effect_size = self.df.iloc[i]['beta_outcome'] / self.df.iloc[i]['beta_exposure']
                    se_ratio = np.sqrt((self.df.iloc[i]['se_outcome']/self.df.iloc[i]['beta_exposure'])**2 +
                                      (self.df.iloc[i]['beta_outcome'] * self.df.iloc[i]['se_exposure'] / self.df.iloc[i]['beta_exposure']**2)**2)
                    precision = 1 / se_ratio if se_ratio > 0 else 0

                    effect_sizes.append(effect_size)
                    precisions.append(precision)
                    colors.append('#1f77b4')

            # 绘制散点
            ax.scatter(effect_sizes, precisions, c=colors, s=80, alpha=0.7,
                      edgecolors='none', linewidth=0, zorder=3)

            # 添加IVW估计参考线 - 根据方法名称匹配真实的IVW结果
            ivw_estimate = 0
            ivw_result = self._find_mr_result_by_method(mr_results, ['inverse', 'variance', 'weighted', 'ivw'])
            if ivw_result:
                ivw_estimate = ivw_result.estimate
            elif mr_results and len(mr_results) > 0:
                # 备用方案：使用第一个结果
                ivw_estimate = mr_results[0].estimate
            elif effect_sizes:
                # 最后备用方案：使用平均值
                ivw_estimate = np.mean(effect_sizes)

            ax.axvline(x=ivw_estimate, color='#d62728', linestyle='-', linewidth=3,
                       alpha=0.8, zorder=2)

            # 添加95%置信区间的漏斗线
            if precisions:
                max_precision = max(precisions)
                precision_range = np.linspace(0.5, max_precision, 100)
                se_range = 1 / precision_range

                upper_ci = ivw_estimate + 1.96 * se_range
                lower_ci = ivw_estimate - 1.96 * se_range

                ax.plot(upper_ci, precision_range, 'k--', alpha=0.6, linewidth=2, zorder=1)
                ax.plot(lower_ci, precision_range, 'k--', alpha=0.6, linewidth=2, zorder=1)

            # 设置标签和标题
            ax.set_xlabel('Causal effect estimate (β)', fontweight='bold')
            ax.set_ylabel('Precision (1/SE)', fontweight='bold')
            ax.set_title('Mendelian Randomization Funnel Plot', fontweight='bold', pad=15)

            # 图例
            legend_elements = [
                plt.Line2D([0], [0], color='#d62728', linewidth=3, label=f'IVW estimate: β = {ivw_estimate:.4f}'),
                plt.Line2D([0], [0], color='black', linestyle='--', linewidth=2, label='95% CI'),
                plt.scatter([], [], c='#1f77b4', s=80, label=f'SNPs (n={len(effect_sizes)})')
            ]

            legend = ax.legend(handles=legend_elements, loc='upper right', frameon=True,
                              framealpha=0.9)
            legend.get_frame().set_facecolor('white')
            legend.get_frame().set_edgecolor('black')
            legend.get_frame().set_linewidth(0.5)

            # 样式优化 - 与案例1完全相同
            ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            plt.tight_layout()
            return self._fig_to_base64(fig)
            
        except Exception as e:
            logger.error(f"Error creating funnel plot: {e}")
            return None
    
    def generate_all_plots(self, mr_results: List[MRMethodResult]) -> dict:
        """
        Generate all visualization plots.
        
        Args:
            mr_results: Results from MR analysis
            
        Returns:
            Dictionary containing base64-encoded plots
        """
        logger.info("Generating all visualization plots")
        
        plots = {
            'scatter_plot': self.create_scatter_plot(mr_results),
            'forest_plot': self.create_forest_plot(mr_results),
            'funnel_plot': self.create_funnel_plot(mr_results)
        }
        
        # Filter out None values
        plots = {k: v for k, v in plots.items() if v is not None}
        
        logger.info(f"Generated {len(plots)} plots successfully")
        return plots
