"""
可视化模块 - 负责生成分析图表
"""
import matplotlib.pyplot as plt
import numpy as np


class VisualizationAnalyzer:
    """可视化分析"""
    
    def __init__(self):
        self.figsize = (30, 24)
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
    
    def plot_basic_analysis(self, df, models_results):
        """绘制基础分析图表"""
        colors = {
            'primary': '#4ECDC4',
            'secondary': '#FF6B6B',
            'accent': '#45B7D1',
            'light': '#F7F7F7',
            'dark': '#2C3E50'
        }
        
        fig, axes = plt.subplots(4, 2, figsize=self.figsize)
        fig.patch.set_facecolor(colors['light'])
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.3, hspace=0.35)
        
        model_names = list(models_results.keys())
        
        # 1. 模型对比柱状图
        ax1 = axes[0, 0]
        outlier_counts = [results['outlier_count'] for results in models_results.values()]
        bars = ax1.bar(model_names, outlier_counts, 
                      color=[colors['primary'], colors['secondary'], colors['accent']][:len(model_names)],
                      edgecolor=colors['dark'], linewidth=1.5)
        ax1.set_title('各模型检测离群值数量', fontsize=16, fontweight='bold')
        ax1.set_ylabel('离群值数量', fontsize=14)
        ax1.grid(axis='y', alpha=0.2, linestyle='--')
        
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height+20,
                    f'{int(height)}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # 2. 训练时间对比
        ax2 = axes[0, 1]
        times = [results['training_time'] for results in models_results.values()]
        bars = ax2.bar(model_names, times, 
                      color=[colors['primary'], colors['secondary'], colors['accent']][:len(model_names)],
                      edgecolor=colors['dark'], linewidth=1.5)
        ax2.set_title('模型训练时间对比', fontsize=16, fontweight='bold')
        ax2.set_ylabel('时间(秒)', fontsize=14)
        ax2.grid(axis='y', alpha=0.2, linestyle='--')
        
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height+0.1,
                    f'{height:.2f}s', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        plt.suptitle('医疗数据离群值检测分析', fontsize=20, fontweight='bold', y=0.98)
        plt.figtext(0.5, 0.94, f'总样本数: {len(df):,} 条', ha='center', fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        
        return fig