# TWLA 量化算法全流程与原理详解

> 基于代码仓库 `TWLA/`（ICML'26）与 NPU 移植实验
> 模型：Qwen3-4B，目标 W1.58A4（权重 1.58-bit + 激活 4-bit）
> 日期：2026-08-19

---

## 〇、总览：TWLA 是什么

TWLA（Ternary Weight Low-bit Activation）是**后训练量化（PTQ）**框架，实现 LLM 的 **W1.58A4**：
- 权重压缩到 **1.58-bit**（三值 {-α, 0, +α} 加每行均值 μ，信息量 log₂3 ≈ 1.58 bit）
- 激活量化到 **4-bit**（可选，W1.58A16 为纯权重量化）

核心思想：**"先整形、再量化、后补偿"**——
1. **KOTMS**：用正交旋转把权重分布"整形"成三模态（三个峰），让三值量化损失最小
2. **E2M-ATQ**：三值化（欧氏初始化 + 流形交替优化）
3. **TRA-GPTQ**：用 GPTQ 的 Hessian 误差补偿，把量化误差从"局部"扩散到"全局最优"
4. **ILA-AMP**：为每层激活分配最优位宽（4-bit 混合精度）

```
原始模型
   │
   ▼ Stage 1: KOTMS 旋转（离线，一次性）
   │  每层 Linear: W → W_rot = C·W·B（C,B 正交），同时记录 L=C, R=B
   ▼ rotated checkpoint
   │
   ▼ Stage 2+3: TRA-GPTQ 逐层量化（离线，校准数据驱动）
   │  每层按 4 组顺序（q/k/v → o → up/gate → down）:
   │    ① 收集激活 Hessian H = XᵀX
   │    ② mask 搜索（结构分组：outlier 2-bit + 正常 1-bit）
   │    ③ E2M-ATQ 三值化（order-2 交替优化）
   │    ④ GPTQ 误差补偿（H⁻¹ 更新剩余列）
   ▼ 量化模型（W1.58）
   │
   ▼ Stage 4: ILA-AMP（可选）
   │  DP 分配每层激活位宽 + LAC 裁剪比校准
   ▼ W1.58A4 最终模型
```

---

## 一、Stage 1：KOTMS —— 正交旋转整形（`scripts/KOTMS.py` + `quantize/k_preprocessor_ternary.py`）

### 1.1 动机

三值量化 {-α, 0, +α} 是"三中心聚类"。若权重分布是任意形态（如高斯/拉普拉斯），三中心聚类损失大；若分布本身就是**三个峰**（三模态），损失最小。KOTMS 的作用就是**用正交变换把任意权重矩阵的分布"掰"成三模态**。

### 1.2 数学原理

对权重 `W[oc, ic]`，把输入维分解 `ic = l × r`（代码 `_pick_lr_factors_budget`：r 取最大满足 `r ≤ √(oc·ic/64)` 且 `ic % r == 0`），构造**两个正交矩阵**：
- `C[l×l]`（左旋转）、`B[r×r]`（右旋转）

旋转：
```
W_rot[oc, l, r] = C · W_2d.reshape(oc, l, r) · B
```

**关键性质（旋转不改变网络输出）**：推理时对激活施加相同旋转
```
x_rot = L @ x @ R   （L = C, R = B）
y = x_rot @ W_rotᵀ = x @ Wᵀ   （正交变换保内积：⟨CxB, CW B⟩ = ⟨x, W⟩）
```
所以旋转后的模型与原模型**数学等价**（代码 `QLinear.forward` 中实现 `x = L @ x @ R`）。

### 1.3 正交矩阵的参数化（Cayley 变换）

直接优化"正交矩阵"困难（约束优化）。代码用**无约束参数化**：
```
S = 任意矩阵（可学习参数）
A = S - Sᵀ           （斜对称）
C = (I - A)⁻¹(I + A)  （Cayley 变换，保证正交）
```
NPU 适配：`linalg.solve` 在 CANN 9.0.0 上 fallback 不稳定，显式在 CPU 上求解（矩阵仅 l×l ≤ ~300，开销可忽略）。

### 1.4 三高斯目标

优化 C、B 使 `W_rot` 分布逼近三高斯混合（等权）：
```
p(w) = ⅓·N(-c, σ₁²) + ⅓·N(0, σ₀²) + ⅓·N(+c, σ₁²)
```
- `c` = 每行 |w| 均值（`c = mean(|x|)`）
- `σ₀ = 0.25·c`、`σ₁ = 0.25·c`（代码 `sigma0_ratio/sigma1_ratio`）
- 损失 = 负对数似然 + `balance_w·‖责任分布 - 1/3‖²`（平衡正则，防止三峰坍缩成一个峰）

优化：Adam，`gmm_iters=100`，lr=1e-2。

### 1.5 为什么同时抑制激活离群值

论文指出：旋转对权重是"整形"，对激活是**统计性抑制离群值**（共享正交变换把激活的能量分散到各维）。这为后续 4-bit 激活量化提供便利。

### 1.6 代码对应

| 代码 | 功能 |
|---|---|
| `TwoSidedOrthoSimple._make_ortho` | Cayley 参数化 → C, B |
| `train_trigauss_two_sided` | 三高斯 GMM 训练循环 |
| `_kronecker_process_trigmm` | 整层权重旋转入口 |
| `scripts/KOTMS.py` | 多卡并行逐层旋转，导出 `{weight, L, R, dim_l, dim_r}` |

### 1.7 实验观察

4B rotated checkpoint 的三模态质量：中心峰占比 0.41（理想 1/3≈0.33）→ 旋转仍有提升空间（调 gmm_iters/sigma/balance_w）。

---

## 二、Stage 2：E2M-ATQ —— 三值化核心（`quantize/E2M_ATQ.py`）

### 2.1 模型假设

每行权重（旋转后）用"非对称三值"近似：
```
order-1:  W_i ≈ μ_i + α_i·T_i     T_i ∈ {-1, 0, +1}ⁿ
order-2:  W_i ≈ μ_i + α⁰_i·T⁰_i + α¹_i·T¹_i   （两个三值基，每元素 9 种组合）
```
- `μ_i`：每行均值（asymmetric 非对称，锚定分布中心）
- `T`：三值符号模板
- `α`：每行缩放（幅值）

### 2.2 交替优化（Euclidean → Manifold 两阶段）

**阶段 A：欧氏初始化（权重域）**——交替迭代：
```
① 固定 (α, μ)，更新 T：Δ = 0.75·mean(|W - μ|)（阈值）
   T_j = +1 若 W_j - μ > Δ；-1 若 < -Δ；否则 0
② 固定 T，更新 (α, μ)：最小二乘闭式解
   α = Σ T·(W-μ) / Σ|T|
   μ ← μ + mean(残差)
```
迭代 `num_iters` 次（代码 10 次，实验证明 15 次无额外收益——已收敛）。

**阶段 B：流形定位（X 域精化）**——冻结 T，用激活自相关 S 加权：
```
目标：最小化输出误差 ‖(W-Ŵ)·Xᵀ‖（而非权重误差）
用 S = X_blockᵀX_block 做加权最小二乘：
  μ = Σ S·(W - αT) / Σ S
  α = Σ S·T·(W - μ) / Σ S·T²
迭代 iter2 次
```
这就是"Euclidean-to-Manifold"：先欧氏空间初始化，再到"输出流形"上精化。

### 2.3 order-2 的 9 候选

order-2 时每元素取值 `μ + α⁰s⁰ + α¹s¹`，`(s⁰,s¹) ∈ {-1,0,1}²` 共 **9 个候选值**，取最近点（`torch.argmin(|target - V|)`）。两个三值基 ≈ 2×1.58 bit 的表示能力，显著降低量化残差。

### 2.4 代码对应

| 函数 | 功能 |
|---|---|
| `ternary_threshold` | Δ 阈值 → T ∈ {-1,0,1} |
| `alpha_from_ternary` | LS 闭式解 α |
| `high_order_residual_alternating_mean_x` | order-2 交替优化（9 候选）+ X 域精化 |
| `Ternarization.quantize` | 统一入口（order 参数切换 1/2） |

### 2.5 实验观察

- E2M 迭代 10 vs 15：MSE 完全相同（10 次收敛，保持 10 省 30% 时间）
- `nanmean → sum/count` 优化：数学等价（mask 均值），NPU 上快 4 倍

---

## 三、Stage 3：TRA-GPTQ —— 结构化三值残差 + GPTQ 补偿（`quantize/tra_gptq.py` + `gptq_fwrd.py`）

### 3.1 整体流程（`gptq_fwrd.twla_fwrd`）

逐层顺序处理（后层看到前层量化后的输出，误差传播）：
```
每层按 4 个 sequential 组：['q,k,v'] → ['o'] → ['up,gate'] → ['down']
每组：
  ① 注册 forward hook，跑 nsamples 个校准样本 → 收集该组每个 Linear 的输入 X、输出
  ② 构建 Hessian H = XᵀX（加权）
  ③ 对每个 Linear 调 fasterquant（mask 搜索 + E2M + GPTQ）
```

### 3.2 Hessian 与 GPTQ 误差补偿原理

**动机**：单纯三值化（RTN）每层误差独立，逐层累积。GPTQ 用二阶信息补偿：
- 量化列 i 后，量化误差 `err = (w_i - q_i)` 会通过权重耦合影响所有未量化列
- 最优补偿（最小化输出误差）：
```
δ = -(w_i - q_i)/[H⁻¹]ᵢᵢ · H⁻¹[i, :]     （H⁻¹ 的第 i 行）
W[:, i:] -= err·Hinv[i, i:]               （只更新未量化列）
```

**Cholesky 数值技巧**：直接存 H⁻¹ 条件数差（cond 大），官方实现存 `U = chol(H⁻¹)`（上三角因子，cond 减半）：
```
Hinv = U，UᵀU = H⁻¹
d = U[i,i]；更新用 U[i,i:]（行 i 的上三角部分）
```

**⚠️ NPU 关键发现（我们的实验）**：`U[i,i:] ≠ Hinv[i,i:]/√Hinv[i,i]`（差 ~7%，因 `Hinv[i,i:] = U[i,i]·U[i,i:] + Σ_{k<i}U[k,i]·U[k,i:]`）——用 Hinv 行替代 U 行会**改变算法语义**（单层 MSE +40%）。修复：CPU fp64 计算 `U = chol(inv(H), upper=True)`。

### 3.3 结构化 mask（outlier 分组）

权重列按重要性分组，不同组用不同阶数：
```
mask 搜索（structural_searching_multip_alternating_group_x）：
  ① top-k 列（按 |W| 重要性）→ order-2 主组（2-bit 结构）
  ② 分位数扫描最优 split（0.10~0.90，41 档）→ 次组
  ③ 剩余列 → order-1 组
最终 4 个 mask：order-2 ×2 + order-1 ×2（num_p=1）
```
即"少数重要列用高精度（两个三值基），多数列用低精度（一个三值基）"。

### 3.4 fasterquant 块循环

```
for 每个块 [col_st, col_ed)（blocksize=128）:
  ① S = X_blockᵀX_block / N（激活自相关，用于 X 域精化）
  ② mask 搜索（hessian 重要性）
  ③ E2M 量化 4 组（q_part_groups）
  ④ 逐列 GPTQ 更新：
     q = Σⱼ q_partⱼ·maskⱼ
     d = Hinv[i,i]
     err = (w - q)/d
     W[:, i:] -= err ⊗ Hinv[i, i:]
  ⑤ 块间传播：W[:, col_ed:] -= Err·Hinv[col_st:col_ed, col_ed:]
```

### 3.5 NPU 适配的数学等价优化（我们的实验验证）

| 优化 | 原理 | 验证 |
|---|---|---|
| H 一次性计算 `H=(2/N)XᵀX` | 逐批加权累积望远镜求和 = 一次性外积和（批大小相同） | MSE 完全一致 |
| S 从 H_raw 取块 `S=H_raw[st:ed,st:ed]·seq/2` | S=(1/N)ΣX_blockᵀX_block，与 H 同源 | MSE 完全一致 |
| `torch.quantile` | np.quantile 的 NPU 原生等价（数值一致） | 0.089s vs CPU 同步 |
| `nanmean → sum/count` | mask 均值等价 | 数值一致 |

---

## 四、Stage 4：ILA-AMP —— 激活混合精度（`scripts/ILA_AMP.py` + `run_twla.py`）

### 4.1 动机

激活全 4-bit 时，少数"弱层"会引发级联退化。ILA-AMP 给每层分配不同位宽（2/4/6/8），在**总位宽预算**下最小化整体损失。

### 4.2 两阶段 NLL 测量（DP cache 构建）

```
order-1：每层单独设 b-bit，其余 8-bit → 测 NLL → delta1[i][b] = NLL(i,b) - NLL_base
order-2：相邻层对 (i-1,i) 设 (bp,b) → 测 NLL → 二阶交互 k[i][bp][b]
```

### 4.3 动态规划分配

```
最小化  Σᵢ delta1[i][bᵢ] + β·Σᵢ k[i][bᵢ₋₁][bᵢ]
约束    Σᵢ bᵢ = avg_bits × L（预算）
```
标准 DP（代码 `dp_allocate_bits_order2`），β=0.5 控制层间交互权重。

### 4.4 LAC（每层激活裁剪比）

激活量化前裁剪：`lac` 控制裁剪比例（0.5~1.0 扫描，选输出 MSE 最小的）。代码 `calibrate_per_layer_lac` 对每个候选位宽校准。

### 4.5 激活量化器

`UniformAffineQuantizer`：per-token 动态、非对称、每层 `lac` 裁剪。推理时 `QLinear.forward` 先旋转激活、再量化、再过线性层。

---

## 五、端到端误差来源分析（结合实验）

| 环节 | 误差来源 | 实验证据 |
|---|---|---|
| KOTMS | 旋转后分布非理想三模态（中心峰 0.41 vs 0.33） | rotated checkpoint 直方图 |
| E2M | 三值化残差（理论下限 ~1.58bit 信息容量） | 单层 MSE ~1.5e3-5e3 |
| GPTQ 补偿 | Hinv 语义/精度（U vs Hinv 差 7%） | **U 语义 MSE -40%**（单层） |
| 逐层累积 | 量化误差沿 36 层传播 | PPL 18.77 vs 原始 ~6-8 |
| 解码 | 重复循环（28/100 题） | GSM8K 错误分析 |

**PPL 与单层 MSE 的关系**：单层 MSE -40% 未转化为端到端 PPL 提升（18.77→19.09 持平）——误差在层间平均化，但 GSM8K（推理能力）对权重精度更敏感，v2 评测待确认。

---

## 六、NPU 移植修改总结（相对上游）

| 修改 | 原因 | 影响 |
|---|---|---|
| `U = chol(inv(H))` CPU fp64 | CANN 9.0.0 Cholesky kernel 同步 bug（507035 崩溃） | 恢复官方语义 + fp64 精度 |
| H 一次性计算 | 128 次大 matmul → 1 次分块外积 | 加速 ~10x（H 收集） |
| S 从 H_raw 取块 | 免每块 matmul | 加速 |
| torch.quantile / nanmean→sum/count | 消除 CPU 同步 / kernel 数 | 加速 30%+ |
| 逐层 checkpoint + supervisor | NPU 环境偶发 SIGSEGV | 断点续跑 |
| Cayley solve 显式 CPU | NPU fallback 崩溃 | 稳定 |
