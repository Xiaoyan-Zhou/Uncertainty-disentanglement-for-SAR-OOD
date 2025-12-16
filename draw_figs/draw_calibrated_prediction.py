# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# calculate the entropy based on calibrated probability
def entropy(p: np.ndarray) -> float:
    # p = np.clip(p, 1e-12, 1.0)
    return float(-np.sum(p * np.log(p + 1e-7)))
 

def draw_calibrated(ours_pre, EDL_pre, save_fig_path):
       
    # 数据
    labels = np.array([0,1,2,3,4,5,6,7,8,9])
    ours_pre = np.array(ours_pre)
    one_branch_pre  = np.array(EDL_pre)

    H_ours = entropy(ours_pre)
    H_one  = entropy(one_branch_pre)

    # 画图
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(labels))
    width = 0.38

    ax.bar(x - width/2, ours_pre, width, label='OURS (dual-branch)')
    ax.bar(x + width/2, one_branch_pre, width, label='EDL (one-branch)')

    ax.set_xlabel('Class label')
    ax.set_ylabel('Calibrated probability')
    ax.set_title(f'Calibrated Prediction per Class  |  Entropy: Ours={H_ours:.3f} nats, One-branch={H_one:.3f} nats')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(ours_pre.max(), one_branch_pre.max()) * 1.15)
    ax.legend(loc='best')
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_fig_path, dpi=300, bbox_inches='tight')
    # 需要矢量图可同时保存 PDF：
    # plt.savefig('calibrated_prediction_grouped.pdf', bbox_inches='tight')
    plt.close(fig)

