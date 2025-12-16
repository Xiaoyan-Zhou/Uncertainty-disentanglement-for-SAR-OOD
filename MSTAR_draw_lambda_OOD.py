import matplotlib.pyplot as plt

# 数据
lambda_values = [0.0001, 0.001, 0.01, 0.05, 0.1, 1.0]
AUROC = [0.95823, 0.98225, 0.99526, 0.99501, 0.9892, 0.65946]
AUROC = [x * 100 for x in AUROC]
AUPR = [0.98573, 0.99429, 0.99836, 0.99835, 0.99592, 0.7583]
AUPR = [x * 100 for x in AUPR]
FPR95 = [0.35518, 0.06422, 0.00972, 0.01644, 0.03635, 1.0]
FPR95 = [x * 100 for x in FPR95]

# 绘图
plt.figure(figsize=(8, 5))

# 绘制 AUROC, AUPR 和 FPR95 曲线
plt.plot(lambda_values, AUROC, label='AUROC', marker='o', linestyle='-', linewidth=4)
plt.plot(lambda_values, AUPR, label='AUPR', marker='o', linestyle='-', linewidth=4)
plt.plot(lambda_values, FPR95, label='FPR95', marker='o', linestyle='-', linewidth=4)

# # 设置对数刻度
plt.xscale('log')

# 设置 x 和 y 轴刻度字体大小
plt.tick_params(axis='x', labelsize=16)  # x轴刻度字体大小
plt.tick_params(axis='y', labelsize=16)  # y轴刻度字体大小

# 图例与标签
plt.xlabel(r'$\lambda$', fontsize=20)
plt.ylabel('Metrics Value', fontsize=20)
# plt.title('Impact of $\lambda$ on AUROC, AUPR, and FPR95', fontsize=14)
plt.legend(fontsize=20, markerscale=1.5)
# plt.grid(True)
plt.tight_layout()

plt.savefig('./results/lambda_OOD_MSTAR.png', dpi=300)
# 显示图表
plt.show()

