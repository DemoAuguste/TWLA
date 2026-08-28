# EdgeRazor Qwen3-1.7B 1.58bit 量化复现 — 进展总结

> 目标：在昇腾 NPU 上复现 EdgeRazor W1.58A8KV8 量化（对标官方 `zhangsq-nju/Qwen3-1.7B-EdgeRazor-1.58bit`，avg 43.91 / BF16 基座 58.65）。
> 方法：QAT（clip 三值化）+ 多级 KD 蒸馏（logits KL + hidden MSE）。

---

## 一、总体进度

| 阶段 | 内容 | 状态 |
|---|---|---|
| Phase 0 | 官方模型验证 | ✅ 完成 |
| Phase 1+3 | 训练脚本 NPU 适配 | ✅ 完成（冒烟通过） |
| Phase 2 | 数据准备 | ✅ 完成（210,483 样本） |
| Phase 4 | 完整训练 | 🔄 进行中（~40%） |
| Phase 5 | 转换 + 评测 + 对比 | ⏳ 待训练完成 |

---

## 二、Phase 0 — 官方模型验证（完成）

- 官方模型在 NPU 上完整跑通：`transformers 5.14` 的 `register_for_auto_class` 缺失，绕过 auto_map，改用 `AutoModelForCausalLM` + 手动 `EdgeRazor.quantize()`。
- **权重结构确认**：QLinear=197, QEmbedding=1；每 256-block 恰好 3 个唯一值（纯三值）。
- **量公式逆向验证**：发布权重整数编码与 `scale=2×mean(|w|)` clip 公式一致度 97-98%。
- **官方模型 ARC-Easy logprob = 59.93%**（正确 letter_ids `[32,33,34,35]`），pred 分布均衡（D443/A266/C255/B224）。

---

## 三、Phase 1+3 — 训练脚本 NPU 适配（完成）

修复 4 个兼容问题：
1. `EdgeRazor(config=...)` 需传 dict（`quant_config_map` 预设）而非字符串路径。
2. example yaml 的 `qwen3attention` target_type 与仓库 v1.3.5 不兼容 → 改用 `[linear, embedding, kv_cache]`。
3. transformers 5.x：`Trainer(tokenizer=)` → `processing_class=`, `torch_dtype` → `dtype`。
4. **NPU `mse_loss` backward dtype bug**（bf16 teacher vs fp32 student hidden）→ 在 trainer 内对齐 dtype。

冒烟测试 30 步通过：显存 8.6G/64G，2.3s/step。

---

## 四、Phase 2 — 数据准备（完成，210,483 样本）

按官方 11.1M 配比同比例缩减至 ~200K：

| 数据源 | 官方配比 | 采样数 | 占比 |
|---|---|---|---|
| ii_7M（Infinity-Instruct 4 域） | 7.45M (67%) | 130,000 | 61.8% |
| ii_gen（Gen） | 1.46M (13%) | 25,000 | 11.9% |
| tulu | 0.61M (5.5%) | 11,000 | 5.2% |
| AM-DeepSeek-R1 | 1.40M (12.6%) | 25,000 | 11.9% |
| task 下游混合（5/8 任务） | 0.24M (2.2%) | 19,483 | 9.3% |

**技术备注**：
- BAAI 原版 gated → 用 manifoldlabs 镜像 parquet 直下。
- AM 文件巨大（14.5GB+），下到 4.3GB（134K 完整条目）后截断采样，数据量充足。
- task 集中 winogrande/social_iqa/openbookqa 因新版 datasets 移除 loading script 失败（缺 3/8）。

---

## 五、Phase 4 — 训练（进行中 ~40%）

### 训练配置（与官方逐一比对）

| 参数 | 官方 Qwen3-1.7B | 我们 |
|---|---|---|
| KD 权重 task/KL/MSE | 0.1 / 2.0 / 0.1 | ✅ 完全一致 |
| 量化配置 | w1_58a8kv8_embint4_qwen3 | ✅ 一致 |
| 总步数 | 2106×2=4212 | 3290（数据 210K×2÷128） |
| lr / scheduler | 2e-5 / constant_with_warmup | ✅ 一致 |
| **有效 batch** | **1536（12×16×8卡）** | **128（4×32×1卡）** ⚠️ |
| **优化器** | adamw_8bit | adamw_torch ⚠️ |
| **attn** | flash_attention_2 | sdpa ⚠️ |

### Loss 轨迹（健康下降）

```
step=   0  total=28.86   ← 初始
step= 500  total=22.79   ← checkpoint-500
step=1000  total=19.28   ← checkpoint-1000
step=1240  total=18.82   ← 当前
```

task CE 15.5→10.2，KL 12.9→8.5，MSE 4.6→6.9。

---

## 六、关键发现 ⚠️ — 中期评测暴露"蒸馏坍缩"

### 评测口径修正

曾误用 `encode(" A")[-1]`（带空格 token id 362/425/356/422），修正为 `encode("A")[0]`（32/33/34/35）。修正后：

| 模型 | ARC-Easy | pred 分布 |
|---|---|---|
| Teacher BF16（同口径自测） | **86.70%** | 均衡 |
| 官方 EdgeRazor 1.58bit | **59.93%** | 均衡（D443/A266/C255/B224） |
| 我们 checkpoint-500 | 33.50% | B 68% |
| 我们 checkpoint-1000 | **24.49%** | **B 97.6%（坍缩）** |
| TEQUILA（旧对照） | ~23% | — |

### 诊断结论（已逐项排除）

1. ✅ 无 NaN/Inf，权重干净。
2. ✅ checkpoint 加载正确（missing=0/unexpected=0）。
3. ✅ **float 权重（未量化）与量化后 ARC 几乎相同**（25.5% vs 24.5%）→ 退化**不是量化引入**，而是 float 权重本身已在蒸馏中坍缩。
4. 根因：**KD/task = 20:1 的高权重下 + 小 batch(128) 梯度噪声大**，student 学到"输出无信息先验（全选 B）"的退化最优解，压低 KL 却无法做选项区分。

### 与官方的关键差异

有效 batch **128 vs 1536（12 倍）**是最大嫌疑。三值化 STE 在小 batch 下梯度震荡严重，导致 logits 先验坍缩。

---

## 七、下一步

- [ ] **两卡 DDP 训练**（扩大有效 batch，缓解坍缩，见方案分析）
- [ ] 训练完成后：权重三值化导出（`_weight_quant(replace_self=True)`）
- [ ] ARC-Easy + HellaSwag 评测，与官方 59.93% 对比
- [ ] 可选：调高 task alpha / label smoothing 强化约束

---
*文档生成时间：训练进行中，checkpoint-1250 附近。*