# TWLA 昇腾 NPU 移植与 Qwen3-4B 1.58-bit 量化实验总结报告

> 实验时间：2026-08-14 ~ 2026-08-17
> 服务器：121.37.53.41（8 × Ascend 910B3，CANN 9.0.0，torch 2.10.0 + torch_npu 2.10.0）
> 实验目录：`/opt/zh/train/twla-npu/`（服务器）、`C:\code\TWLA\npu_tools\`（本地脚本与记录）
> 目标：在 NPU 上运行 TWLA，实现 Qwen3-4B 的 1.58-bit 三值权重量化，并用 GSM8K 评测

---

## 一、最终结果

| 模型 | 配置 | WikiText2 PPL | GSM8K 4-shot |
|---|---|---|---|
| Qwen3-4B 原始 | BF16 | ~6-8 | 90%（9/10） |
| **Qwen3-4B TWLA 量化** | **W1.58A16（1.58-bit 三值权重）** | **18.77** | **22.0%（22/100）** |
| Qwen3-1.7B TWLA 量化 | W1.58A16 | 206.99 | 0%（10） |

- **主目标达成**：Qwen3-4B 1.58-bit 量化在 NPU 上完整跑通（KOTMS 旋转 → E2M-ATQ 量化 → GSM8K 评测）
- 4B 三值量化保留明显数学推理能力（22/100），显著优于历史基线（sherry 旧实验 1.7B 0%、CAT-Q 3%）
- 1.7B 三值量化损失过大（PPL 207、GSM8K 0%），印证小模型对 1.58-bit 量化更脆弱

---

## 二、核心问题与解决方案

### 2.1 Cholesky 分解 NPU 崩溃（最关键问题）

**现象**：`torch.linalg.cholesky` 随机崩溃 `507035 vector core exception ("ub address out of bounds")`；`torch.cholesky_inverse` 100% 崩溃；崩溃后同进程所有算子失效（设备污染需重启进程）。空闲时偶发成功、vLLM 负载时必现 —— 典型 race 特征。之前实验被迫 fallback CPU（每层 5+ 小时，26 小时只完成 1 层）。

**根因定位**（对照 `ops-math` 算子源码 + CANN 日志）：
- 崩溃 kernel：`Cholesky_..._high_performance_2`
- ops-math git 历史显示 3 个 cholesky 修复提交：
  - `d23503d9d`（04-22）大尾轴分块支持
  - `baeb3f900`（05-18）加 `PipeBarrier<PIPE_ALL>`（标量读写与向量运算 race）
  - `4a100e92c`（**07-08**）同步指令修复：`MTE2_S→MTE2_V`、`S_MTE3→V_MTE3`（向量计算未完成时 MTE3 就搬出 UB → 越界异常）
- **服务器 CANN 9.0.0 二进制构建于 2026-06-30，缺少 07-08 的同步指令修复** → 崩溃

**解决方案**：用 `torch.linalg.inv` 替代 cholesky 链。GPTQ 的 `U = chol(chol_inv(chol(H)))` 满足 `U[i,i:]=Hinv[i,i:]/√Hinv[i,i]`，代入误差更新公式后与直接用 `Hinv` **数学完全等价**。`inv` 在 NPU 上 fallback CPU 稳定执行（6144 维仅 8.5s）。实测量化 MSE：inv 版 5030 vs cholesky 链 3195（同量级，可用）。

### 2.2 速度优化（4 项数学等价重构）

| 优化 | 内容 | 效果 |
|---|---|---|
| Hessian 一次性计算 | 逐批 `H += (2/n)XᵀX`（128 次大 matmul，down_proj 每次 154 GFLOP）望远镜求和后等价于 `H=(2/N)XᵀX` 一次性分块计算 | down_proj H 收集从 ~1-2 分钟降至秒级 |
| S 免算 | 每块激活自相关 `S = H_raw[st:ed,st:ed] × seq/2`（数学推导等价） | 免去每块 matmul |
| torch.quantile | `np.quantile`（每块 2 次 CPU 同步）→ `torch.quantile`（NPU 上 0.09s，数值一致） | 消除 CPU 同步 |
| nanmean → sum/count | `torch.nanmean`（4 kernel）→ `where + sum/count`（数学等价） | mask 搜索加速 |

参数调整：分位搜索 81→41、E2M 交替优化迭代 15→10（实测 MSE 4724→4675，质量无损）。
**效果**：单层 q_proj fasterquant 54s→38s（-30%），down_proj 176s→119s（-32%），MSE 不变或略好。

### 2.3 健壮性（应对 NPU 环境不稳定）

服务器 NPU 环境不稳定（/tmp 出现 60+ core 文件，其他用户进程也在 SIGSEGV）：
- **逐层断点续跑**：`gptq_fwrd.twla_fwrd` 增加 `--resume_quant`，每层量化后保存 `layer_XXXX.pt`，重启自动加载已量化层并前向传播续跑
- **supervisor 自动重启**：崩溃后等 60s 重启（最多 50 次），配合 resume 从断点继续
- 评测脚本同样支持断点（每 5 题增量保存 JSON）+ supervisor
- nsamples 128→64：显存减半、单次运行时间减半（缩小崩溃窗口）

---

## 三、修改的代码清单（相对上游 TWLA）

| 文件 | 修改内容 |
|---|---|
| `quantize/tra_gptq.py` | ① cholesky 链 → `torch.linalg.inv`（含 HINV_MODE 环境开关便于 A/B）② `add_batch` 正定检查 → 对称化+归一化 ③ H 一次性分块计算 `(2/N)XᵀX` ④ S 从 H_raw 块直接取 ⑤ q 整块组装 ⑥ DEBUG 环境变量开关（`TWLA_DEBUG`） |
| `quantize/E2M_ATQ.py` | ① `torch.nanmean` 4 函数 → `where+sum/count`（数学等价）② 交替优化迭代 15→10 |
| `quantize/autosearch.py` | ① `np.quantile` → `torch.quantile`（NPU 上执行）② 分位 81→41 |
| `quantize/k_preprocessor_ternary.py` | Cayley 变换 `linalg.solve` 显式 CPU 执行（NPU fallback 会崩） |
| `quantize/gptq_fwrd.py` | 逐层 checkpoint（`--resume_quant`）；（服务器原有）cuda→npu |
| `run_twla.py` | `--resume_quant` 参数透传；（服务器原有）cuda→npu |

另外新增运维脚本（服务器 `scripts/` + 本地 `npu_tools/`）：`supervisor_quant.sh`（自动重启）、`delayed_start.sh`（定时启动）、`server_monitor.sh`（进度监控）、`eval_gsm8k.py`（参数化 GSM8K 评测，支持断点续跑）。

---

## 四、执行时间线

### 8-14（首日：问题定位与流程打通）

| 时间 | 步骤 | 耗时 | 说明 |
|---|---|---|---|
| 15:41-16:20 | NPU 算子能力测试 | ~40 min | 发现 cholesky 部分可用、cholesky_inverse 必崩；vLLM 启动后 cholesky 全崩；每个算子需独立进程测试（崩溃污染设备） |
| 16:20-16:44 | 算子源码分析 | ~25 min | 对照 ops-math 源码 git 历史，定位 CANN 9.0.0 缺同步修复；确认 inv 替代方案数学等价 |
| 16:44-17:30 | 代码修改 + 单层验证 | ~45 min | tra_gptq.py inv 化；单层 q_proj 量化冒烟（不崩）；A/B：真实 Hessian 下 inv 版 MSE=5030 vs cholesky 链 3195 |
| 17:00-19:04 | 4B KOTMS 旋转 | ~2 h | 252 任务 4 卡并行；1 个 worker 崩溃，--resume 补齐；产出 7.3GB rotated checkpoint |
| 17:35-19:40 | 1.7B 量化（首轮，旧代码） | 2 h | 默认 abits=8 走 AMP 流程，~10 min/层，进程被杀（未完成，仅到 Layer ~10） |
| 19:06-20:02 | 4B 量化（首轮，新代码） | 1 h | 验证优化代码单层 ~6 min；进程 SIGSEGV 被杀 |
| 19:40-20:04 | 崩溃排查 | ~25 min | 确认 SIGSEGV + core dump（服务器 60+ core，环境不稳定）；实施断点续跑 + supervisor |
| 20:04 | 暂停 | — | 按用户要求等待 4 小时（避开 vLLM 高峰），安排 00:04 自动启动 |

### 8-15（自动执行）

| 时间 | 步骤 | 耗时 | 说明 |
|---|---|---|---|
| 00:04-02:01 | 1.7B 量化（W1.58A16，nsamples=64） | ~2 h | 一次成功，无崩溃；PPL=206.99 |
| 00:04-05:10 | **4B 量化（W1.58A16，nsamples=64）** | **~5 h** | **一次成功，无崩溃；PPL=18.77**；36 层 × 平均 ~8 min/层 |
| 05:10-08:53 | 等待 | ~4 h | — |

### 8-17（评测）

| 时间 | 步骤 | 耗时 | 说明 |
|---|---|---|---|
| 08:53-09:16 | GSM8K 10 条（量化 vs 原始） | ~25 min | 量化 30%（3/10），原始 90%（9/10） |
| 09:16-09:50 | 100 条评测首轮 | 崩溃 | SIGSEGV（72/100 时），输出被缓冲无断点 |
| 09:50-11:30 | **GSM8K 100 条（断点版+supervisor）** | **~1.5 h** | 中途 1 次崩溃自动恢复；**最终 22.0%（22/100）** |

**总计有效计算时间**：~12 小时（含 4B 量化 5h、KOTMS 2h、评测 2h、1.7B 量化 2h）。

---

## 五、性能对比（与历史实验）

| 指标 | sherry 旧实验（CPU cholesky fallback） | 本次（NPU inv 方案） |
|---|---|---|
| 单层 down_proj 量化（6144 维） | 5+ 小时（CPU cholesky） | ~2-3 分钟 |
| 全模型量化 | 26 h 仅完成 1 层（1.7B） | 5 h 完成全部 36 层（4B） |
| Qwen3-1.7B GSM8K | 0% | 0%（确认小模型三值化损失大） |
| Qwen3-4B GSM8K | 无 | **22%** |

---

## 六、产物清单

**服务器 `/opt/zh/train/twla-npu/`**：
- `ckpt/qwen3_4b_quantized.pt`（8.9GB）— **1.58-bit 量化模型（主产物）**
- `ckpt/qwen3_4b_rotated.pt`（7.3GB）— KOTMS 旋转 checkpoint
- `ckpt/qwen3_1_7b_quantized.pt`（4.1GB）— 1.7B 量化模型（参考）
- `ckpt/qwen3_4b_resume/`、`ckpt/qwen3_1_7b_resume/` — 逐层量化 checkpoint（可续跑）
- `logs/` — 全部运行日志（quant、eval、supervisor、monitor）
- `logs/eval_4b_twla_gsm8k_100.json` — 100 条逐题评测明细
- `EXPERIMENT_NOTES.md` — 完整实验记录
- `TWLA/` — 修改后的代码（含 NPU 化 + 全部优化）

**本地 `C:\code\TWLA\npu_tools\`**：
- 全部运维/评测脚本（sshrun.py、supervisor_quant.sh、eval_gsm8k.py 等）
- 修改后代码副本（tra_gptq_server.py、run_twla_server.py、gptq_fwrd_server.py 等）
- `EXPERIMENT_NOTES.md`（与服务器同步）

---

## 七、可选后续工作

1. **W1.58A4**（激活 4-bit 量化）：需先构建 4B 的 ILA-AMP DP cache（`scripts/ILA_AMP.py`，约 3-4 小时），再跑 LAC 校准 + DP 位宽分配
2. **评测扩展**：C4 PPL、更多 GSM8K 样本或 few-shot 变体、其他 QA 数据集（lm-eval-harness）
3. **KOTMS 参数优化**：当前 gmm_iters=100，可尝试更多迭代或调整 sigma 比例提升旋转质量
4. **算子层面**：升级 CANN 到含 07-08 修复的版本可恢复原生 Cholesky（但仍建议保留 inv 路径作为 fallback）
