import numpy as np
import matplotlib.pyplot as plt

# 数据
domains = [
    "MSTAR(clean)",
    "lowres(0.5m)", "lowres(1.0m)",
    "speckle(0.7)", "speckle(0.8)", "speckle(0.9)", "speckle(1.0)",
    "defocus(0.007)", "defocus(0.005)", "defocus(0.003)", "defocus(0.001)"
]

ours = np.array([99.07, 54.57, 54.65, 96.88, 97.69, 98.11, 98.35, 48.71, 56.34, 75.28, 95.13])
edl  = np.array([95.40, 49.61, 49.59, 92.78, 93.81, 94.03, 94.25, 38.06, 47.26, 66.58, 88.73])

# 画图
x = np.arange(len(domains))
width = 0.38  # 柱宽

fig, ax = plt.subplots(figsize=(12, 5))
bars1 = ax.bar(x - width/2, ours, width, label='OURS (dual-branch)')
bars2 = ax.bar(x + width/2, edl,  width, label='EDL (one-branch)')

ax.set_ylabel('ACC (%)')
ax.set_title('OOD Generalization: ACC on Degraded SAR Domains')
ax.set_xticks(x)
ax.set_xticklabels(domains, rotation=30, ha='right')
ax.set_ylim(0, 100)
# ax.legend()
ax.legend(loc='best', bbox_to_anchor=(1.02, 1.0), borderaxespad=0)  
ax.grid(axis='y', linestyle='--', alpha=0.5)

# 在柱顶显示数值（保留两位小数）
def autolabel(rects):
    for r in rects:
        h = r.get_height()
        ax.annotate(f'{h:.2f}',
                    xy=(r.get_x() + r.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

# autolabel(bars1)
# autolabel(bars2)

fig.tight_layout()
plt.savefig('ood_generalization_bar.png', dpi=300)
plt.show()
