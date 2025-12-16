# draw OOD detection performance under different parameter lambda
import matplotlib.pyplot as plt

# ---- Data ----
lambdas = [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0]

# AUROC (%)
auroc = {
    "SAMPLE (OOD)":      [88.04, 95.09, 99.31, 91.71, 90.97, 47.21],
    "FUSAR-ship (OOD)":  [85.74, 91.79, 95.09, 88.53, 85.96, 43.87],
    "SAR-ACD (OOD)":     [90.32, 95.90, 99.89, 93.34, 95.32, 35.00],
    # "Average":           [88.03, 94.26, 98.10, 91.19, 90.75, 42.03],
}

# FPR95 (%)
fpr = {
    "SAMPLE (OOD)":      [68.84, 34.49, 0.24, 35.46, 67.58, 99.76],
    "FUSAR-ship (OOD)":  [84.02, 53.21, 25.26, 55.35, 75.41, 96.88],
    "SAR-ACD (OOD)":     [67.03, 33.88, 0.42, 33.83, 38.10, 96.95],
    # "Average":           [73.30, 40.53, 8.64, 41.54, 60.36, 97.86],
}

# ---- Plot AUROC ----
plt.figure(figsize=(7, 4))
for name, vals in auroc.items():
    plt.plot(lambdas, vals, marker='o', label=name)

plt.xscale('log')
plt.xlabel(r'$\lambda$')
plt.ylabel('AUROC (%)')
plt.title('Effect of $\lambda$ on OOD Detection (AUROC)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('draw_figs/lambda_auroc.png', dpi=300)
plt.show()

# ---- Plot FPR95 ----
plt.figure(figsize=(7, 4))
for name, vals in fpr.items():
    plt.plot(lambdas, vals, marker='o', label=name)

plt.xscale('log')
plt.xlabel(r'$\lambda$')
plt.ylabel('FPR (%)')
plt.title('Effect of $\lambda$ on OOD Detection (FPR95)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('draw_figs/lambda_fpr95.png', dpi=300)
plt.show()
