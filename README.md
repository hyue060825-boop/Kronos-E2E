# Kronos + E2E：面向风险偏好的端到端 A 股选股因子学习框架

本目录是从原项目中整理出的 GitHub 展示包，聚焦“端到端学习 + 因子优化层 + Markowitz 组合约束”的核心实验。原始工作区中的大体量缓存、模型权重、Qlib 数据与第三方源码副本没有放入本包，避免仓库过大。

## 实验结论

在加入交易成本、做空限制、持仓约束、最大权重约束与 L1 换手惩罚后，风险偏好居中的 `eta=2.5` 取得本组实验最高的 CSI300 年化超额收益：

| strategy                   | annualized_excess_return | information_ratio | max_drawdown |
| -------------------------- | -----------------------: | ----------------: | -----------: |
| SiteE2E-MDScaledMW-eta=0.5 |                  3.3082% |            0.3834 |    -10.5156% |
| SiteE2E-MDScaledMW-eta=1   |                  3.3085% |            0.3835 |    -10.5154% |
| SiteE2E-MDScaledMW-eta=2.5 |        **3.3099%** |  **0.3836** |    -10.5154% |
| SiteE2E-MDScaledMW-eta=5   |                  1.7135% |            0.2015 |    -10.2349% |
| SiteE2E-MDScaledMW-eta=10  |                 -5.1165% |           -0.7234 |    -10.3599% |

主图见 [figures/Method1_SiteE2E_MeanDownsideScaledMarkowitz.png](figures/Method1_SiteE2E_MeanDownsideScaledMarkowitz.png)。

## 方法概述

核心信号采用 Kronos 预测路径构造的均值收益和下行风险：

```text
core_signal = cs_zscore(mean) - 0.3 * cs_zscore(downside)
```

训练阶段将因子缩放和残差适配器纳入端到端优化，为不同风险偏好 `eta` 学习一组因子参数；回测阶段使用 scaled Markowitz 组合优化，并施加实际交易约束。

主要设置：

| 参数             | 值                       |
| ---------------- | ------------------------ |
| etas             | 0.5, 1.0, 2.5, 5.0, 10.0 |
| top_n            | 50                       |
| cov_lookback     | 60                       |
| rebalance_freq   | 5                        |
| turnover_penalty | 0.02                     |
| max_weight       | 0.04                     |
| mu_scale         | 0.003                    |
| epochs           | 30                       |
| qp_steps         | 5                        |
| projection_steps | 8                        |

## 目录结构

```text
Kronos-E2E/
  src/
    finetune/      # 端到端训练、因子适配器、Markowitz/Qlib 回测入口
    model/         # Kronos 模型结构与推理依赖
  results/
    metrics/       # 回测指标、alpha 参数、mu_scale 扫描结果
    reports/       # 实验报告
  figures/         # 论文/README 可直接引用的结果图
  logs/            # 训练和参数扫描日志，便于追溯
```

## 复现实验说明

本包保留了关键源码与结果，适合作为项目展示和结果追溯。完整复现还需要本地 Qlib A 股数据、Kronos 预训练权重或已缓存预测文件，因这些文件体量较大，建议在 README 中说明获取方式，而不是直接提交到 GitHub。

入口脚本：

```bash
cd github_experiment_package/src/finetune
python train_site_e2e_mean_downside_scaled_markowitz.py
```

如果从零开始运行，需要先配置 `config.py` 中的数据、模型和输出路径，并准备 Qlib 数据。
