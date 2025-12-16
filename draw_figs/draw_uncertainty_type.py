import numpy as np
import matplotlib.pyplot as plt

# 定义函数（类似正弦，可以加点扰动）
def func(x):
    return np.sin(x) + 0.1*np.sin(3*x)

# 定义不同区间和采样范围（最少点数，最多点数）
# 增加一个参数：是否加噪声
intervals = [
    (0, np.pi/2, 100, 150, False),    # 0 到 π/2 之间，随机采样，噪声小
    (np.pi/2, np.pi, 50, 100, True),   # π/2 到 π 之间，随机采样，噪声大
    (np.pi, 2*np.pi, 80, 120, False)  # π 到 2π 之间，随机采样，噪声小
]

x_vals = []
y_vals = []

# 在各区间生成随机数量的点
for start, end, min_n, max_n, noisy in intervals:
    n_points = np.random.randint(min_n, max_n)   # 在范围内随机决定点数
    x = np.linspace(start, end, n_points)
    y = func(x)
    if noisy:
        y += np.random.normal(0, 0.2, size=n_points)  # 在特定区间加入高斯噪声
    x_vals.append(x)
    y_vals.append(y)

# 拼接
x_all = np.concatenate(x_vals)
y_all = np.concatenate(y_vals)

# 只用于散点：过滤掉 x > 3 的点
mask = x_all <= 3
x_scatter = x_all[mask]
y_scatter = y_all[mask]

# （可选）打乱散点顺序
shuffle_idx = np.random.permutation(len(x_scatter))
x_scatter = x_scatter[shuffle_idx]
y_scatter = y_scatter[shuffle_idx]

# 可视化
plt.figure(figsize=(8, 4))

# 1) 画散点（仅 x ≤ 3）
plt.scatter(x_scatter, y_scatter, s=10, c="blue", alpha=0.6, label="Sampled Points (x ≤ 5)")

# 2) 画底层函数曲线（全范围 0 ~ 2π）
x_curve = np.linspace(0, 1.7*np.pi, 400)
plt.plot(x_curve, func(x_curve), 'r--', label="Underlying Function (0–2π)")

# plt.legend()
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Random Non-uniform Sampling with Local Noise (hide samples for x > 5)")

# 保存图像
plt.savefig("nonuniform_sampling_with_noise_hide_x_gt_5.png", dpi=300, bbox_inches="tight")
plt.close()
