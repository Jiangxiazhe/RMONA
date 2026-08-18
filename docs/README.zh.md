# RMONA — Riemannian-MONA: 流形上的曲率感知加速优化器

> 本文件是 README.md 的中文版（英文版为仓库主文档）。

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)]()
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)]()

**Riemannian-MONA (RMONA)** 是一个面向**正交约束优化**的 PyTorch 优化器——在 Stiefel 流形
(半正交矩阵 `{W : WᵀW = I}`) 上组合了 **SMP 精确方向求解**（谱符号/谱范数最陡下降）
与 **EMA 梯度差曲率通道**（平行移动的切空间传递），填补 2×2 研究版图的唯一空位：

| | 无曲率通道 | 有曲率通道（梯度差） |
|---|---|---|
| 欧氏空间 | Muon (Jordan 2024) ✓ | MONA (2026) ✓ |
| **Stiefel 流形** | Skewon / Manifold-Muon ✓ | **Riemannian-MONA（本文）★** |

核心科学问题：梯度差在流形上如何合法定义？→ **平行移动 + 重投影近似**（零额外成本）。

---

## 亮点

- **硬约束保证**：流形方法训练全程 `‖WᵀW − I‖_F ≤ 1e-5`（欧氏 Muon/MONA 约束崩溃到 28）。
- **曲率通道增益**：pMNIST 正交 RNN 上 RMONA 显著优于无曲率的 Manifold-Muon（90.6% vs 89.5%，3 seed 一致）。
- **模块化设计**：方向求解（SMP 闭式/对偶/交替投影）、曲率通道（α、β_a）、retraction（QR/Cayley）均可独立替换，天然支持消融。
- **零依赖流形库**：纯 PyTorch 实现，不依赖 geoopt；支持任意维度参数混合分组（流形参数 + AdamW 参数同优化器）。

## 实验效果

### 实验 2：pMNIST 正交 RNN（病态长程任务）

Permuted MNIST 序列分类，单隐层 tanh RNN（`W_hh ∈ O(128)`），
**12000 步 × lr 扫描 × 3 seeds**。下表为各方法取最优 lr 后的 3-seed 均值：

| 方法 | 空间 | 最优测试准确率 | 约束违反 ‖WᵀW − I‖_F |
|---|---|---|---|
| `cayley`（Cayley SGD，任务专用） | Stiefel | **92.39%** | 1.5e-5 |
| `cayleyRNN`（参数化，任务专用） | — | **92.28%** | 7e-6 |
| **`rmona`（本文）** | Stiefel | **90.63%** | 4e-6 |
| `manifold_muon`（无曲率通道） | Stiefel | 89.47% | 3e-6 |
| `expRNN`（参数化） | — | 87.60% | 4e-5 |

核心结论：

- **曲率通道贡献成立**：`rmona` 与其唯一区别是 EMA 梯度差曲率通道的直接对手
  `manifold_muon`，在同一任务与配置下 **+1.2 个点（90.63% vs 89.47%）**，
  3 个 seed 结果一致。
- **硬正交约束全程保持**：所有流形方法 `‖WᵀW − I‖_F ≤ 1.5e-5`，
  而欧氏 Muon/MONA 约束崩溃到 orth ≈ 28。
- 任务专用方法（`cayley`/`cayleyRNN`）仍是 pMNIST SOTA；RMONA 面向
  **通用正交约束优化**，是通用流形优化器中表现最强的。

训练曲线与最终准确率：

![Exp2 pMNIST: 测试准确率与训练损失](figures/exp2_pmnist.png)

![Exp2 pMNIST: 最优测试准确率（best-lr，3 seeds 均值）](figures/exp2_acc.png)

### 实验 1：凸问题正确性验证（正交矩阵学习）

Rayleigh-Ritz 特征问题，3 seeds × lr 扫描，2000 步。所有流形方法均收敛到
解析最优；RMONA 400 步收敛且保持正交（1.2e-6），与最快的基线持平。

![Exp1 Rayleigh-Ritz: loss 与约束曲线](figures/exp1_rr_curves.png)

![Exp1 Rayleigh-Ritz: 收敛速度](figures/exp1_rr_steps.png)

## 快速开始

```bash
pip install -e .
# 运行单元测试（64 项，含流形原语与优化器约束保持）
pytest tests/
```

在你的模型中使用（循环权重 `W_hh` 保持正交，其余参数用 AdamW）：

```python
import torch
from rmona import FlowOptimizer

model = MyOrthogonalRNN()   # 含 W_hh: nn.Parameter, W_hh 初始化为正交矩阵
opt = FlowOptimizer([
    {'params': [model.W_hh], 'method': 'rmona',
     'lr': 0.005, 'momentum': 0.9, 'alpha': 0.1, 'beta_a': 0.9,
     'smp_iter': 8, 'smp_path': 'closed', 'retraction': 'qr'},
    {'params': [p for n, p in model.named_parameters() if n != 'W_hh'],
     'method': 'adamw', 'lr': 1e-3},
])

for x, y in loader:
    opt.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    opt.step()
```

## 支持的优化方法

`FlowOptimizer` 统一 11 种方法，按参数组选择：

| method | 空间 | 方向求解 | 曲率通道 | 说明 |
|---|---|---|---|---|
| `rmona` | Stiefel | SMP 闭式解 | EMA 梯度差 | **本文算法** |
| `manifold_muon` | Stiefel | SMP 对偶迭代 | — | Bernstein 2025 |
| `skewon` | Stiefel | SMP 交替投影 | — | Solonko et al. 2026 |
| `cayley` | Stiefel | 切空间 SGD | — | Cayley SGD (Li et al. 2020) |
| `rsgd` / `rsgd_m` | Stiefel | 切空间 SGD | — | Riemannian SGD (±动量) |
| `radam` | Stiefel | 切空间 Adam | — | Riemannian Adam |
| `muon` / `mona` | 欧氏 | NS 正交化 | mona 有 | 无约束对照（约束崩溃） |
| `adamw` | 任意 | AdamW | — | 非流形参数 |

## 实验复现

```bash
# Exp1: 正交矩阵学习（凸问题正确性验证，~6 min）
python examples/exp1_matrix.py --task rr     # Rayleigh-Ritz
python examples/exp1_matrix.py --task proc   # Procrustes

# Exp2: pMNIST 正交 RNN（病态长程任务，12000 步 × lr 扫描 × 3 seeds）
CUDA_VISIBLE_DEVICES=0 python examples/exp2_pmnist.py \
    --steps 12000 --hidden 128 --lr_grid 0.005,0.01,0.02 \
    --param_lr_grid 0.001 --seeds 0 1 2

# 绘图
python examples/plot_results.py
```

多卡并行见 `scripts/run_exp2_4gpu.sh`（--resume 支持断点续跑）。

## 文档

- [design.md](design.md) — 算法设计（背景、SMP 闭式解推导、曲率通道动机、理论分析思路）
- [experiments.md](experiments.md) — 实验设置与结果（Exp1 正确性 / Exp2 pMNIST 对比）
- [api.md](api.md) — API 参考

## 引用

```bibtex
@misc{rmona2026,
  title  = {Riemannian-MONA: Curvature-Aware Manifold Optimization for Orthogonal Constraints},
  author = {RMONA contributors},
  year   = {2026},
  note   = {https://github.com/Jiangxiazhe/RMONA}
}
```

## License

MIT License，见 [LICENSE](../LICENSE)。
