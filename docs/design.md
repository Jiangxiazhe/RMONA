# Riemannian-MONA：Stiefel 流形上的曲率感知加速优化算法

**算法设计与分析文档 v0.2（流形化迭代版）**

> 本文档是 **Riemannian-MONA（RMONA）**——面向正交约束优化的流形曲率感知加速优化器——的完整设计文档。算法定位为 2×2 研究版图（参数空间 × 时间维曲率通道）中唯一空位象限：**Stiefel 流形约束 × 梯度差曲率加速通道**（MONA 是欧氏的，Skewon 无曲率通道，两者交叉点无人做）。
>
> 本文档为 RMONA 的完整设计文档。理论部分为证明思路（proof sketch）级别，完整严格证明与实验数据待后续版本补充。



---

## 目录

1. [算法背景介绍](#1-算法背景介绍)
2. [相关工作介绍](#2-相关工作介绍)
3. [算法方法](#3-算法方法)
4. [理论分析证明思路](#4-理论分析证明思路)
5. [算法实现与实验对比方案](#5-算法实现与实验对比方案)
6. [与现有算法的对比](#6-与现有算法的对比)
7. [本文算法的贡献](#7-本文算法的贡献)
8. [参考文献](#参考文献)

---

## 1. 算法背景介绍

### 1.1 从 Muon 的流形本质说起

Muon（Jordan et al., 2024）的更新 $O = \text{NewtonSchulz}(M) \approx UV^\top$ 在几何上等价于求解线性最小化预言（LMO）：

$$
\min_{B} \langle M, B \rangle \quad \text{s.t.} \quad \|B\|_2 \le 1
$$

其解为 $M$ 的负极分解因子 $-UV^\top$（$M = U\Sigma V^\top$）。已有工作（shen2025convergence）进一步证明：**方阵情形下 Muon 等价于 Stiefel 流形上的自然梯度下降**，即 Muon 执行的是谱范数约束下的最陡下降。这意味着 Muon 天然是"流形感知"的——只是其参数 $W$ 停留在欧氏空间，仅更新方向被投影到正交矩阵流形。

### 1.2 正交约束优化与 Stiefel 流形

Stiefel 流形定义为半正交矩阵集合：

$$
\mathrm{St}(n,p) = \{ X \in \mathbb{R}^{n \times p} : X^\top X = I_p \}, \quad n > p
$$

正交约束（$W^\top W = I$）广泛存在于深度学习中：正交 RNN（防止梯度爆炸/消失）、QK 正交正则的 Transformer、度量学习、谱归一化、正交卷积、图匹配等。约束流形优化有成熟工具（geoopt、Riemannian SGD/Adam），但存在两大痛点：

1. **动量与加速难以迁移**：流形上动量的正确传递需要**平行移动**（parallel transport），实现复杂且计算昂贵；
2. **大规模场景效率低**：经典流形优化器（RSGD/RAdam、Cayley SGD）在深度模型规模下收敛慢、超参敏感。

### 1.3 Muon 的流形化：已有进展与遗留问题

**Skewon（arXiv 2608.06218, 2026-08）**&#x7ED9;出关键结果：Muon 在 Stiefel 流形上的切空间 LMO（Stiefel Muon Problem, SMP）存在**精确闭式解**：

$$
\min_{B} \langle M, B \rangle \quad \text{s.t.} \quad \|B\|_2 \le 1,\; B \in T_X \mathrm{St}(n,p)
$$

其中 $T_X \mathrm{St} = \{B : X^\top B + B^\top X = 0\}$。Bernstein（Modula docs, 2025）用 Lagrangian 对偶给出另一求解路径（解形如 $-msign(G + 2W(\Lambda + \Lambda^\top))$）。这些工作解决了"**流形上选方向**"的问题，但**均未引入时间维曲率加速**——动量仍是朴素一阶动量。

### 1.4 曲率加速的动机（MONA 的欧氏答案）

MONA（arXiv 2605.26842, 2026-05）在欧氏 Muon 上引入 EMA 梯度差加速通道：

$$
a_k = \beta_a a_{k-1} + (1-\beta_a)(g_k - g_{k-1}), \qquad g'_k = g_k + \alpha a_k
$$

其理论动机：$g_k - g_{k-1} \approx -2 H_k g_k \approx -\nabla \|\nabla f\|^2$，即梯度差指向**远离尖锐极小值**的方向。该通道在 1B–68B MoE 预训练中显著优于 Muon 与 AdamW。**但 MONA 完全工作在欧氏空间**，其梯度差在流形上不合法（切空间随点变化）。

### 1.5 本文目标与定位

**Riemannian-MONA = Skewon 的方向求解（SMP 闭式解）× MONA 的曲率加速（梯度差通道）× 平行移动的切空间传递**。填补 2×2 版图的空位象限：

|            | 无曲率通道                    | 有曲率通道（梯度差）               |
| ---------- | ------------------------ | ------------------------ |
| 欧氏空间       | Muon（Jordan 2024）✓       | MONA（2026）✓   |
| Stiefel 流形 | Skewon / Manifold-Muon ✓ | **Riemannian-MONA（本文）★ 空位** |

**核心科学问题**：梯度差在 Stiefel 流形上如何合法定义？平行移动的近似误差有多大？曲率加速能否与谱约束方向求解相容？

---

## 2. 相关工作介绍

### 2.1 欧氏优化器谱系（背景）

- **动量与加速**：重球（Polyak 1964）、NAG（Nesterov 1983）、Su–Boyd–Candès 连续时间 ODE 分析；
- **自适应**：Adam/AdamW、Adan（NME 梯度差通道，ICLR 2023）、Sophia（对角 Hessian）；
- **矩阵感知**：Shampoo（Kronecker 预条件）、Muon（NS 正交化）。

### 2.2 Muon 家族变体（欧氏）

| 工作                | 机制                            | 状态           |
| ----------------- | ----------------------------- | ------------ |
| Muon（Jordan 2024） | NS 正交化                        | 基础           |
| Moonlight（2025）   | WD + per-param scale + ZeRO-1 | 规模化          |
| MONA（2026-05）     | EMA 梯度差曲率加速                   | 欧氏，1B–68B 验证 |
| DynMuon（2026-05）  | 谱形变 $U\Sigma^p V^\top$ 动态 p   | 谱维度          |
| Mousse（2026-03）   | Shampoo 式 Kronecker 预条件       | 空间曲率         |

### 2.3 流形优化

- **Riemannian SGD/Adam**（geoopt, Kochurov et al. 2020）：切空间投影 + retraction，动量需平行移动；
- **Cayley SGD**（Li et al. 2020）：Cayley 变换保持正交性，用于正交 RNN；
- **结构化流形方法**：Stiefel/Grassmann 上的 QR/极分解 retraction（Absil et al., 2008）。

### 2.4 Stiefel 上的 Muon（流形化已有工作）

| 工作                                    | 内容                               | 与本文关系                     |
| ------------------------------------- | -------------------------------- | ------------------------- |
| **Skewon**（arXiv 2608.06218, 2026-08） | SMP 精确闭式解 + 非凸一阶收敛保证             | **方向求解的基础**，本文直接采用其 SMP 解 |
| **Bernstein**（Modula docs, 2025）      | 立题 manifold Muon，Lagrangian 对偶求解 | 理论奠基，闭式解另一路径              |
| Cesista / 苏剑林                         | 交替投影 / 不动点迭代求解                   | 求解方案                      |
| ARS 系列                                | Fisher × Stiefel 分解 + SRM 谱残留监测  | 工程先例（含自适应阻尼思想）            |
| shen2025convergence                   | 方阵 Muon = Stiefel 自然梯度           | 解释性基础                     |

**关键差异**：以上流形工作全部为"无曲率加速通道"的一阶动量；本文是**首个在 Stiefel 约束优化中引入梯度差曲率加速**的工作。

### 2.5 控制理论视角（保留联系）

反步框架（Chen, Liu & Xu, arXiv 2606.27722）的"增广 z 通道"与曲率阻尼项 $k_1 \nabla^2 f \dot{x}$ 为梯度差通道提供了控制论解释；在流形设定下，该通道对应"切空间上的曲率修正"，反步框架的耗散结构可迁移到流形 Lyapunov 分析（见 §4）。

---

## 3. 算法方法

### 3.1 问题设定

考虑正交约束优化：

$$
\min_{W \in \mathrm{St}(n,p)} f(W), \qquad f(W) = \mathbb{E}_\zeta[\ell(W; \zeta)]
$$

在 $W$ 处，欧氏梯度 $g = \nabla f(W)$ 需投影到切空间：

$$
\mathrm{Proj}_{T_W \mathrm{St}}(g) = g - \frac{1}{2} W(W^\top g + g^\top W)
$$

### 3.2 关键设计决策

| 决策点     | 选择                                                                                      | 依据                        |
| ------- | --------------------------------------------------------------------------------------- | ------------------------- |
| 切空间投影   | 标准 Stiefel 投影 $g - \frac{1}{2}W(W^\top g + g^\top W)$                                   | 保持更新切向流形                  |
| 梯度差的定义  | **平行移动后差分**：$D_k = G_k - \mathcal{P}_{k \to k-1}(G_{k-1})$                              | 切空间随点变化，直接差分不合法（核心设计）     |
| 平行移动的实现 | **重投影近似**（默认）：$G_{k-1} \mapsto \mathrm{Proj}_{T_{W_k}}(G_{k-1})$；完整版用测地平行移动             | 重投影一阶相容、零额外成本；误差界见 Thm C' |
| 曲率通道    | MONA 式 EMA：$A_k = \beta_a A_{k-1} + (1-\beta_a) D_k$                                    | 平滑抑制切空间噪声                 |
| 方向求解    | **Skewon SMP 闭式解**：$\min\langle M,B\rangle$ s.t. $\|B\|_2 \le 1, B \in T_W \mathrm{St}$ | 谱约束方向 + 切空间约束，比单纯 NS 更精确  |
| 回流形     | QR retraction（默认）/ Cayley retraction                                                    | 标准选择，一阶相容                 |
| 曲率注入位置  | 动量前串联（MONA 式）：$M_k = \mu M_{k-1} + (G_k + \alpha A_k)$                                  | 与 MONA 保持一致的工程结论          |
| 权重衰减    | 流形约束下**不适用**（$W^\top W = I$ 已限范数）；若需正则化改用谱正则项                                           | 与欧氏 Moonlight 的 WD 定位不同   |

### 3.3 伪代码（Riemannian-MONA）

```
输入: 学习率 η, 动量 μ, 加速系数 α, 加速记忆 βa,
      SMP 迭代步数 T, retraction R（QR/Cayley）

初始化: W₀ ∈ St(n,p), M₀ = 0, A₀ = 0, G_prev = 0

for k = 1, 2, ... do
    # ① 欧氏梯度 + 切空间投影
    g = ∇f(W)  （对 2D 正交约束参数；其余参数走 AdamW）
    G = g − ½·W·(Wᵀg + gᵀW)

    # ② 曲率通道（平行移动 + 差分）
    G_prev_T = Proj_{T_W}(G_prev)          # 重投影近似平行移动
    D = G − G_prev_T
    A = βa·A + (1−βa)·D                    # EMA 加速缓冲（切空间向量）

    # ③ 曲率感知动量（MONA 式串联）
    M = μ·M + (G + α·A)

    # ④ 方向求解（Skewon SMP 闭式解）
    O = SolveSMP(M, W)                     # min⟨M,B⟩ s.t. ‖B‖₂≤1, B∈T_W St

    # ⑤ 更新 + retraction 回流形
    W = Retract(W, η·O)                    # QR: qf(W + η·O)

    # ⑥ 保存切向量供下一步差分
    G_prev = G
end for
```

### 3.4 SMP 闭式解的工程实现

Skewon 证明 SMP 存在精确闭式解；Bernstein 的对偶路径给出迭代式：

$$
A(\Lambda) = -msign(M + 2W(\Lambda + \Lambda^\top)), \qquad X^\top A + A^\top X = 0
$$

工程实现二选一：

- **直接调用 Skewon 闭式解公式**（若作者开源实现可用）；
- **NS 迭代 + 对偶修正**（Bernstein 方案）：对 $M + 2W(\Lambda + \Lambda^\top)$ 做 NS 谱符号，内层解对偶变量 $\Lambda$ 使切空间约束满足（不动点迭代，通常数步收敛）。

### 3.5 超参数与默认值

| 超参数        | 默认值           | 说明              |
| ---------- | ------------- | --------------- |
| 学习率 η      | 0.01–0.05     | 流形问题通常小于欧氏 Muon |
| 动量 μ       | 0.9           | 流形动量（配合重投影近似）   |
| 加速系数 α     | 0.1（0.05–0.3） | 需消融             |
| 加速记忆 βa    | 0.9           | 同 MONA          |
| SMP 迭代 T   | 5–10          | NS 类迭代          |
| retraction | QR（默认）        | Cayley 备选       |
| 非约束参数      | AdamW 混合      | 同 Muon 惯例       |

### 3.6 与 Skewon / MONA 的差异

|        | Skewon  | MONA    | **Riemannian-MONA**         |
| ------ | ------- | ------- | ------------------------ |
| 参数空间   | Stiefel | 欧氏      | **Stiefel**              |
| 曲率加速通道 | 无       | EMA 梯度差 | **EMA 梯度差（平行移动版）**       |
| 方向求解   | SMP 闭式解 | NS 正交化  | **SMP 闭式解**              |
| 动量传递   | 朴素（重投影） | 欧氏      | **重投影 + 差分一致性处理**        |
| 收敛分析   | 非凸一阶    | 非凸逃逸    | **非凸一阶 + 平行移动误差界（本文新增）** |

---

## 4. 理论分析证明思路

> 本节为 proof sketch。核心新贡献：**平行移动近似（重投影）的误差界（Thm C'）**，以及流形设定下曲率通道的收敛框架（Thm A'/B'/D'）。

### 4.1 假设（流形版本）

**假设 M1（retraction 一阶相容）**：$R_W(\xi)$ 满足 $R_W(0) = W$，$DR_W(0)[\xi] = \xi$（标准假设，QR/Cayley retraction 均满足）。  
**假设 M2（光滑性）**：$f$ 在流形邻域内 $L$-光滑（沿 retraction 方向：$\langle \nabla f \circ R_W(\xi) - \nabla f(W), \xi' \rangle$ 类 Lipschitz 条件）。  
**假设 M3（曲率有界）**：$\|\nabla^2 f(W)\| \le \rho$（流形局部）。  
**假设 M4（随机梯度）**：$g_k = \nabla f(W_k) + \xi_k$，$\mathbb{E}\|\xi_k\|^2 \le \sigma^2$。

### 4.2 Theorem A'：连续时间收敛（流形 ODE + 曲率阻尼）

沿 retraction 的连续化更新对应流形上的二阶 ODE：

$$
\nabla^{\mathrm{Riem}}_{\dot W} \dot W + c_1 \dot W + c_2 \mathrm{Proj}_{T_W}(\nabla f(W)) + c_3 \mathrm{Proj}_{T_W}(\nabla^2 f(W)[\dot W]) = 0
$$

其中 $\nabla^{\mathrm{Riem}}$ 为流形仿射联络（协变导数）。

**Theorem A'（流形连续收敛）**：设 M1–M3 成立，$f$ 在流形上测地凸（geodesically convex），则：

1. 曲率阻尼项（$c_3$）贡献额外耗散 $\propto -c_3 \|\mathrm{Proj}_{T_W}(\nabla f)\|^2$，加速收敛；
2. 最优解的收敛速率随曲率感知强度改善，且**在测地凸性下保持全局收敛**。

**证明思路**：在切空间构造能量函数 $\mathcal{E}(W, \dot W)$，利用流形版本的"梯度-协变导数交换恒等式"（$\frac{d}{dt}\|\nabla f\|^2 = 2\langle \nabla^2 f[\dot W], \nabla f \rangle +$ 联络修正项），将曲率项转化为显式耗散。测地凸性给出 $\langle \nabla f(W), \mathrm{Exp}_W^{-1}(W^*) \rangle \ge f(W) - f^*$ 的流形版本，用于锁定收敛目标。

### 4.3 Theorem B'：离散收敛（切空间 Lyapunov）

**Theorem B'（离散收敛）**：设 M1–M4 成立，$\eta \le \eta_{\max}(\mu, \alpha, \beta_a, L, \rho)$，则：

- 流形上 $f(W_k) - f^* \to 0$（次线性 $O(1/k)$；测地强凸下线性收敛待严格化）。

**证明思路**：构造切空间 Lyapunov $\Phi_k = f(W_k) - f^* + \frac{1}{2}\|M_k\|^2 + \frac{\alpha\delta}{2}\|A_k\|^2 + \tau\langle \mathrm{Proj}_{T_{W_k}}\nabla f(W_k), M_k \rangle$。关键步骤：

1. 用 retraction 一阶相容性（M1）把 $f(R_W(\eta O))$ 展开为 $f(W) - \eta\langle \nabla f, O \rangle + O(\eta^2)$；
2. 利用 SMP 解的性质：$O$ 与 $M$ 的夹角余弦 $\ge \sigma_{\min}(M)/\sigma_{\max}(M)$（谱比下界，继承 Theorem C 思路）；
3. 平行移动重投影的误差吸收进 $O(\eta^2 \rho)$ 项（见 Thm C'）；
4. 选择系数使二次型负定，得 $\Phi$ 递减。

### 4.4 Theorem C'：平行移动近似误差界（本文核心理论贡献）

**问题**：真平行移动 $\mathcal{P}_{k \to k-1}: T_{W_{k-1}} \to T_{W_k}$ 计算昂贵；本文默认用**重投影** $\mathrm{Proj}_{T_{W_k}}(G_{k-1})$ 近似。

**Theorem C'（重投影误差界）**：设 M1–M3 成立，$\|W_k - W_{k-1}\|_F = O(\eta)$，则对任意切向量 $v \in T_{W_{k-1}}\mathrm{St}$：

$$
\left\| \mathrm{Proj}_{T_{W_k}}(v) - \mathcal{P}_{k\to k-1}(v) \right\|_F \le C \cdot \rho \cdot \eta \cdot \|v\|_F
$$

其中 $C$ 依赖流形曲率（Stiefel 流形截面曲率上界）。**一阶相容性**：误差 $O(\eta)$，不破坏 Theorem B' 的主阶收敛。

**证明思路**：利用 Stiefel 流形平行移动的微分方程（沿测地线的协变导数），将重投影与真平行移动的差展开为曲率项（Riemann 曲率张量）的一阶贡献；$\|W_k - W_{k-1}\| = O(\eta)$ 来自 retraction 一阶相容性与步长。**意义**：为"忽略平行移动、用重投影"的工程选择提供了严格依据——误差随步长线性收缩，且在 $\eta$ 主导的收敛分析中不改变主阶。

**推论（何时需要真平行移动）**：当流形曲率 $\rho$ 大、步长 $\eta$ 大（加速早期）或精度要求高时，误差不可忽略——此时切换为测地平行移动（数值求解协变导数方程）或缩短步长。这给出"重投影 vs 真平行移动"的切换判据。

### 4.5 Theorem D'：切空间噪声方差压缩

**Theorem D'**：设 M4 成立。平滑曲率通道 $A_k$ 的稳态方差满足

$$
\mathbb{E}\|A_k\|^2 \le \frac{1-\beta_a}{1+\beta_a} \cdot 2\sigma^2 + \text{（确定性曲率项）}
$$

（$\beta_a = 0.9$ 时噪声压缩至约 5%）。证明与欧氏版（Thm D）一致，仅需将范数限制在切空间子空间。

### 4.6 理论边界与开放问题

| 开放问题          | 现状与计划                                                    |
| ------------- | -------------------------------------------------------- |
| 测地强凸下的线性收敛速率  | 待严格化（需流形上的 strong geodesic convexity 与协变 Lyapunov 的精细分析） |
| 平行移动的精确实现成本   | 重投影（$O(np^2)$）vs 数值平行移动（$O(np^2)$ 每步，但实现复杂）的权衡待实验验证      |
| 非凸逃逸分析        | MONA 的逃逸尖锐极小值论证可迁移到流形（需 Lipschitz Hessian + 流形曲率假设）      |
| 曲率通道与谱约束的相互作用 | SMP 解会重新谱整形方向，曲率信息保留程度待数值验证（风险预案：必要时谱压缩而非全压 1） |

---

## 5. 算法实现与实验对比方案

### 5.1 实现要点（PyTorch 风格）

```python
import torch
import geoopt  # 或手写 Stiefel 操作

def proj_tangent(W, g):
    """Stiefel 切空间投影: g − ½W(Wᵀg + gᵀW)"""
    return g - 0.5 * W @ (W.T @ g + g.T @ W)

def parallel_transport_approx(W_prev, W_cur, v):
    """重投影近似平行移动"""
    return proj_tangent(W_cur, v)   # 一阶相容，误差 O(η)（Thm C'）

def solve_smp(M, W, T=5):
    """Skewon SMP 闭式解（Bernstein 对偶迭代路径）:
       A(Λ) = −msign(M + 2W(Λ+Λᵀ)), 内层不动点满足切空间约束"""
    Λ = torch.zeros_like(W @ W.T)
    for _ in range(T):
        X = M + 2 * W @ (Λ + Λ.T)
        A = -msign_newtonschulz(X)     # NS 谱符号
        Λ = Λ - 0.5 * (W.T @ A + A.T @ W)  # 不动点修正（示意）
    return A

def qr_retraction(W, xi):
    """QR retraction: qf(W + ξ)"""
    Q, _ = torch.linalg.qr(W + xi)
    return Q

def stiefel_mona_step(W, g, st, η=0.02, μ=0.9, α=0.1, βa=0.9):
    G = proj_tangent(W, g)                        # ① 切空间投影
    G_prev_T = parallel_transport_approx(W_prev, W, st['G_prev'])
    D = G - G_prev_T                              # ② 平行移动后差分
    st['A'] = βa * st['A'] + (1 - βa) * D
    st['M'] = μ * st['M'] + (G + α * st['A'])     # ③ 曲率感知动量
    O = solve_smp(st['M'], W)                     # ④ 方向求解
    W_new = qr_retraction(W, η * O)               # ⑤ 回流形
    st['G_prev'] = G
    return W_new
```

### 5.2 内存与计算分析

- **状态**：动量 M + 加速缓冲 A + 上一步切向量 G_prev = 3 个切空间矩阵（约 3×参数量，可比 AdamW；其中 A、G_prev 可 bf16 压缩）；
- **计算**：切空间投影 $O(np^2)$、SMP 求解 $O(T \cdot np^2)$（NS 类）、QR retraction $O(np^2)$——总开销与 Muon 同阶，重投影平行移动**零额外成本**；
- **通信**：与 Muon 相同（权重 all-gather），分布式沿用 Moonlight ZeRO-1 方案。

### 5.3 实验协议

**实验 1：正交约束基准（小规模，验证正确性）**

- 任务：正交矩阵学习（$W^\top W = I$ 下的最小二乘/特征问题）、CIFAR 上的正交卷积（orthoConv 正则）；
- 基线：Riemannian SGD / Riemannian Adam（geoopt）、Cayley SGD、Skewon（无曲率通道）、Manifold-Muon（Bernstein 闭式解）；
- 指标：约束违反 $\|W^\top W - I\|_F$、收敛步数、最终损失。

**实验 2：正交 RNN / 深度正交网络**

- 任务：顺序 MNIST / pMNIST（orthogonal RNN 经典基准）；
- 基线：expRNN、cayleyRNN（Li et al. 2020）、Skewon、RSGD；
- 目的：验证"流形曲率加速"在长程记忆任务上的收益（梯度差通道应改善病态时间尺度）。

**实验 3：正交约束 Transformer（大规模，核心卖点）**

- 任务：小型 GPT 上对 QK 投影施加正交正则/硬约束（$Q^\top Q = I$），或使用正交注意力变体；
- 基线：AdamW（软正则）、Skewon、Manifold-Muon、MONA（欧氏，不可行对照）；
- 目的：验证大规模可行性 + 与欧氏 MONA 的对比（流形约束 + 曲率加速的组合增益）。

### 5.4 消融矩阵

| 编号    | 配置                                       | 目的                 |
| ----- | ---------------------------------------- | ------------------ |
| B0    | Riemannian-MONA 完整（α=0.1, βa=0.9）           | 主配置                |
| B1    | α=0（退化为 Skewon）                          | 验证曲率通道净收益          |
| B2    | 不用重投影，直接切空间差分（非法操作对照）                    | 验证平行移动必要性（预期劣化）    |
| B3    | 真平行移动（数值求解）vs 重投影                        | 验证 Thm C' 误差界与切换判据 |
| B4–B6 | α ∈ {0.05, 0.2, 0.3} × βa ∈ {0.85, 0.95} | 超参敏感性              |
| B7    | QR vs Cayley retraction                  | retraction 选择影响    |

### 5.5 评估指标

1. **约束保持**：$\|W^\top W - I\|_F$（流形方法的核心质量指标）；
2. **收敛性**：损失/函数值下降速率；
3. **稳定性**：梯度范数尖峰、切向量范数轨迹；
4. **效率**：wall-clock、显存；
5. **任务性能**：pMNIST 准确率、Transformer 验证 loss / 下游指标。

### 5.6 预期结果与分析框架

- **强预期**：B0 vs B1 在病态/长程任务（orthoRNN）上曲率通道收益显著；B2 性能劣化（验证平行移动必要）；B3 中重投影与真平行移动在 $\eta$ 小时无显著差异（验证 Thm C'）；
- **风险预案**：若曲率信息被 SMP 谱约束抹平 → 转向"谱压缩而非全压 1"（部分谱整形），或把曲率通道移到 SMP 之后（结构对比实验）；
- **与欧氏 MONA 的关系**：不直接竞争——MONA 优化无约束欧氏权重，RMONA 优化正交约束权重；对比点是"约束 + 曲率"的组合增益是否大于各自单独收益之和。

---

## 6. 与现有算法的对比

### 6.1 机制对比

| 特性     | RSGD/RAdam | Cayley SGD | Skewon  | Manifold-Muon（Bernstein） | MONA（欧氏）   | **Riemannian-MONA**   |
| ------ | ---------- | ---------- | ------- | ------------------------ | ---------- | ------------------ |
| 参数空间   | Stiefel    | Stiefel    | Stiefel | Stiefel                  | 欧氏         | **Stiefel**        |
| 曲率加速通道 | ✗          | ✗          | ✗       | ✗                        | ✓（EMA 梯度差） | **✓（平行移动版）**       |
| 方向求解   | 切空间 SGD    | Cayley 变换  | SMP 闭式解 | 对偶 + msign               | NS 正交化     | **SMP 闭式解**        |
| 动量传递   | 平行移动       | 无/朴素       | 重投影     | 重投影                      | 欧氏         | **重投影（有误差界）**      |
| 收敛保证   | 有（经典）      | 有          | 非凸一阶    | 少                        | 非凸逃逸       | **非凸一阶 + 平行移动误差界** |
| 大规模友好  | 中          | 中          | 高       | 高                        | 高          | 高                  |

### 6.2 复杂度对比（每步、每参数）

| 算法               | 计算          | 内存               | 平行移动         |
| ---------------- | ----------- | ---------------- | ------------ |
| RSGD/RAdam       | $O(np^2)$   | 2×参数             | 需（真平行移动昂贵）   |
| Cayley SGD       | $O(np^2)$   | 1–2×             | 免（Cayley 结构） |
| Skewon           | $O(T np^2)$ | 1×               | 重投影          |
| MONA（欧氏）         | $O(np^2)$   | 3×               | 不适用          |
| **Riemannian-MONA** | $O(T np^2)$ | 3×（A/G_prev 可压缩） | **重投影（零额外）** |

### 6.3 适用场景

- **Riemannian-MONA 最优场景**：显式正交约束的深度模型（orthoRNN、正交 Transformer 层、度量学习、谱归一化），尤其是病态曲率 + 长程依赖任务；
- **不适用场景**：无正交约束的一般训练（应直接用 MONA/Muon）；
- **与 MONA 定位区分**：MONA 解决"欧氏权重 + 曲率加速"，RMONA 解决"约束权重 + 曲率加速"——互补而非竞争。

---

## 7. 本文算法的贡献

### 7.1 理论贡献

1. **平行移动重投影误差界（Theorem C'）**：首次定量刻画"用重投影近似平行移动"的误差 $O(\eta \rho)$，为流形优化中普遍使用的重投影技巧提供严格依据，并给出"何时需要真平行移动"的切换判据；
2. **流形曲率通道收敛框架（Theorem A'/B'）**：将梯度差曲率加速（MONA）推广到测地凸流形设定，证明曲率项转化为显式耗散；
3. **切空间噪声方差定理（Theorem D'）**：平滑通道的方差压缩因子 $(1-\beta_a)/(1+\beta_a)$ 在流形设定下成立。

### 7.2 算法与工程贡献

1. **首个"流形约束 × 曲率加速"优化器**：填补 2×2 版图唯一空位象限（Skewon 无曲率、MONA 无约束）；
2. **切空间梯度差的正确定义**：平行移动 + 差分（重投影实现零成本），解决"流形上梯度差不合法"的根本问题；
3. **模块化设计**：SMP 方向求解（Skewon）、曲率通道（MONA）、重投影平行移动（本文）三者可独立替换，天然支持消融与迭代。

### 7.3 局限与未来工作

1. **测地强凸线性收敛未严格化**；
2. **非凸逃逸分析未迁移到流形**（MONA 的逃逸论证 + 流形曲率假设）；
3. **大规模实证未执行**（实验 3）；
4. **真平行移动的高效实现**（数值协变导数 vs 结构近似）作为精度优先方案的备选；
5. **谱压缩 retraction 族**：当曲率信息被全谱整形抹平时，引入 $U\Sigma^p V^\top$ 谱形变（连接 DynMuon）作为方向求解的推广。

### 7.4 查新结论与定位（2026-08-11）

- **已占领地**：Muon（欧氏·无曲率）、MONA（欧氏·有曲率）、Skewon/Bernstein/Manifold-Muon（Stiefel·无曲率）、DynMuon（谱形变）、Mousse（空间曲率预处理）、ARS 系列（Fisher×Stiefel）；
- **本工作空位**：Stiefel 约束 + 时间维曲率加速（平行移动处理切空间差分）——查新时未发现任何已发表工作；
- **风险提示**：该圈子（Bernstein、苏剑林、Cesista、Skewon 团队）极小且极活跃，需快速推进；保持差异化护城河 = 平行移动误差分析（Thm C'）+ 大规模正交约束实验。

---

## 参考文献

1. Nesterov, Y. *A method of solving a convex programming problem with convergence rate O(1/k²)*. Soviet Mathematics Doklady, 1983.
2. Polyak, B. T. *Some methods of speeding up the convergence of iteration methods*. 1964.
3. Kingma, D. P., Ba, J. *Adam: A Method for Stochastic Optimization*. ICLR 2015.
4. Loshchilov, I., Hutter, F. *Decoupled Weight Decay Regularization*. ICLR 2019.
5. Xie, X., et al. *Adan: Adaptive Nesterov Momentum Algorithm for Faster Optimizing Deep Models*. ICLR 2023.
6. Jordan, K., et al. *Muon: An optimizer for hidden layers in neural networks*. 2024.
7. Liu, J., et al. (Moonshot AI). *Muon is Scalable for LLM Training*（Moonlight）. arXiv:2502.16982, 2025.
8. Xu, H., et al. *MONA: Muon Optimizer with Nesterov Acceleration for Scalable Language Model Training*. arXiv:2605.26842, 2026.
9. Solonko, M., Molozhavenko, A., Rakhuba, M. *Muon on the Stiefel Manifold Admits an Exact Closed-Form Update*（Skewon）. arXiv:2608.06218, 2026.
10. Bernstein, J. *The Modula Docs: Manifold Muon（谱范数下 Stiefel 最陡下降的 Lagrangian 对偶求解）*. 2025.
11. Cesista, F. L.; Su, J. *Manifold Muon 的交替投影 / 不动点迭代解*. 2025.
12. *DynMuon: A Dynamic Spectral Shaping View of Muon*. arXiv:2605.17109, 2026.
13. *Mousse: Rectifying the Geometry of Muon with Curvature-Aware Preconditioning*. arXiv:2603.09697, 2026.
14. Chen, S., Liu, J., Xu, C. *A Backstepping Framework for Unconstrained Accelerated Optimization Algorithms*. arXiv:2606.27722, 2026.
15. Absil, P.-A., Mahony, R., Sepulchre, R. *Optimization Algorithms on Matrix Manifolds*. Princeton Univ. Press, 2008.
16. Kochurov, M., et al. *Geoopt: Riemannian Optimization in PyTorch*. arXiv:2005.02819, 2020.
17. Li, J., et al. *Optimization on Stiefel Manifolds via Cayley Transform*. ICLR 2020.
18. Anil, R., et al. *Scalable Second Order Optimization for Deep Learning*（Shampoo）. arXiv:1802.09568, 2018.

---

*文档版本 v0.2（流形化迭代）· 2026-08-11 · 理论部分为证明思路级别，实验部分为方案设计，待执行*
