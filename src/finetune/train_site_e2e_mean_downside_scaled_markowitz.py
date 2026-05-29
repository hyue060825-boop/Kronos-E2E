import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

import qlib
from qlib.config import REG_CN

sys.path.append("../")
from config import Config
from method1_return_adapter import load_or_generate_split_payloads, payload_to_return_features
from qlib_markowitz_test import (
    MarkowitzConfig,
    RebalanceWeightStrategy,
    build_markowitz_weight_frame,
    cs_zscore,
    load_cached_predictions,
    load_close_prices,
    plot_comparison,
    run_single_backtest,
    signal_frame_to_series,
)
from train_site_e2e_adapter import (
    SiteE2EConfig,
    build_site_samples,
    dataframe_to_markdown,
    feature_dates,
    parse_etas,
)
from train_site_e2e_mean_downside_adapter import (
    MeanDownsideCoreConfig,
    score_from_adapter,
    standardize_features,
    train_adapter_for_eta,
)


def build_cached_prediction_features(config: Config) -> dict[str, pd.DataFrame]:
    prediction_dfs = load_cached_predictions(config)
    required = {"last", "mean", "max", "min"}
    missing = required - set(prediction_dfs)
    if missing:
        raise KeyError(f"Missing cached prediction signals: {', '.join(sorted(missing))}")

    return {
        "mean_ret": cs_zscore(prediction_dfs["mean"]),
        "last_ret": cs_zscore(prediction_dfs["last"]),
        "max_ret": cs_zscore(prediction_dfs["max"]),
        "downside_risk": cs_zscore((-prediction_dfs["min"]).clip(lower=0.0)),
        "path_volatility": cs_zscore((prediction_dfs["max"] - prediction_dfs["min"]).abs()),
    }


def write_report(
    path: str,
    alpha_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    figure_path: str,
    split_features: dict[str, dict[str, pd.DataFrame]],
    site_cfg: SiteE2EConfig,
    core_cfg: MeanDownsideCoreConfig,
    mu_scale: float,
):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Method 1 Mean-Downside Core + Scaled Markowitz\n\n")
        f.write(
            "本实验保留旧版最强信号 `mean_downside_ret` 的按天全市场 z-score 构造，"
            "训练阶段只微调核心信号的缩放与小残差；最终回测阶段复用 "
            "`qlib_markowitz_test.py` 中的 scaled Markowitz 逻辑，即 `mu_scale` 与 L1 换手惩罚。\n\n"
        )
        f.write("## 数据切分\n\n")
        for split in ("train", "val", "test"):
            dates = feature_dates(split_features[split])
            f.write(f"- `{split}_data.pkl`: {dates.min().date()} 至 {dates.max().date()}，共 {len(dates)} 个信号日\n")
        f.write("\n## 训练后的参数\n\n")
        f.write(dataframe_to_markdown(alpha_df, index=False))
        f.write("\n\n## Qlib 测试集回测结果\n\n")
        f.write(dataframe_to_markdown(metrics_df, index=True))
        f.write("\n\n## 结果图\n\n")
        f.write(f"![Mean-Downside Core Scaled Markowitz]({os.path.relpath(figure_path, os.path.dirname(path))})\n\n")
        f.write("## 参数\n\n")
        f.write(f"- core_signal: `cs_zscore(mean) - {core_cfg.downside_coef} * cs_zscore(downside)`\n")
        f.write(f"- residual_scale: {core_cfg.residual_scale}\n")
        f.write(f"- core_scale_range: +/-{core_cfg.core_scale_range}\n")
        f.write(f"- etas: {site_cfg.etas}\n")
        f.write(f"- top_n: {site_cfg.top_n}\n")
        f.write(f"- cov_lookback: {site_cfg.cov_lookback}\n")
        f.write(f"- rebalance_freq: {site_cfg.rebalance_freq}\n")
        f.write(f"- turnover_penalty: {site_cfg.gamma}\n")
        f.write(f"- max_weight: {site_cfg.max_weight}\n")
        f.write(f"- mu_scale: {mu_scale}\n")
        f.write(f"- epochs: {site_cfg.epochs}\n")
        f.write(f"- qp_steps: {site_cfg.qp_steps}\n")
        f.write(f"- projection_steps: {site_cfg.projection_steps}\n")


def main():
    parser = argparse.ArgumentParser(description="Mean-downside initialized Site-E2E with scaled Markowitz backtest.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force_inference", action="store_true")
    parser.add_argument("--etas", default="0.5,1.0,2.5,5.0,10.0")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch_days", type=int, default=20)
    parser.add_argument("--top_n", type=int, default=50)
    parser.add_argument("--cov_lookback", type=int, default=60)
    parser.add_argument("--gamma", type=float, default=0.02)
    parser.add_argument("--max_weight", type=float, default=0.04)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha_l2", type=float, default=1e-3)
    parser.add_argument("--turnover_tau", type=float, default=0.001)
    parser.add_argument("--qp_steps", type=int, default=5)
    parser.add_argument("--qp_lr", type=float, default=0.05)
    parser.add_argument("--projection_steps", type=int, default=8)
    parser.add_argument("--downside_coef", type=float, default=0.3)
    parser.add_argument("--residual_scale", type=float, default=0.2)
    parser.add_argument("--core_scale_range", type=float, default=0.5)
    parser.add_argument("--mu_scale", type=float, default=0.003)
    parser.add_argument(
        "--train_rebalance_only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max_train_days", type=int, default=None)
    parser.add_argument("--max_val_days", type=int, default=None)
    args = parser.parse_args()

    kronos_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(kronos_root)

    base_config = Config()
    site_cfg = SiteE2EConfig(
        etas=parse_etas(args.etas),
        epochs=args.epochs,
        patience=args.patience,
        batch_days=args.batch_days,
        top_n=args.top_n,
        cov_lookback=args.cov_lookback,
        gamma=args.gamma,
        max_weight=args.max_weight,
        lr=args.lr,
        alpha_l2=args.alpha_l2,
        turnover_tau=args.turnover_tau,
        qp_steps=args.qp_steps,
        qp_lr=args.qp_lr,
        projection_steps=args.projection_steps,
    )
    core_cfg = MeanDownsideCoreConfig(
        downside_coef=args.downside_coef,
        residual_scale=args.residual_scale,
        core_scale_range=args.core_scale_range,
        use_cs_zscore=True,
    )

    qlib.init(provider_uri=base_config.qlib_data_path, region=REG_CN)
    payloads = load_or_generate_split_payloads(base_config, args.device, force=args.force_inference)

    result_dir = os.path.join(base_config.backtest_result_path, base_config.backtest_save_folder_name)
    os.makedirs(result_dir, exist_ok=True)
    split_features = {}
    close_dfs = {}
    for split, payload in payloads.items():
        feature_path = os.path.join(result_dir, f"site_e2e_md_scaled_return_features_{split}.pkl")
        raw_features = payload_to_return_features(payload, feature_path)
        split_features[split] = standardize_features(raw_features, core_cfg)
        dates = feature_dates(split_features[split])
        instruments = list(split_features[split]["mean_ret"].columns)
        history_start = pd.Timestamp(dates.min()) - pd.Timedelta(days=max(180, site_cfg.cov_lookback * 3))
        close_dfs[split] = load_close_prices(instruments, start_time=history_start, end_time=dates.max())

    train_instruments = list(split_features["train"]["mean_ret"].columns)
    val_instruments = list(split_features["val"]["mean_ret"].columns)
    train_samples = build_site_samples(
        split_features["train"],
        close_dfs["train"],
        train_instruments,
        site_cfg,
        max_days=args.max_train_days,
    )
    val_samples = build_site_samples(
        split_features["val"],
        close_dfs["val"],
        val_instruments,
        site_cfg,
        max_days=args.max_val_days,
    )
    if args.train_rebalance_only:
        train_samples = train_samples[:: site_cfg.rebalance_freq]
        val_samples = val_samples[:: site_cfg.rebalance_freq]
        print(
            f"Using rebalance-only samples: train={len(train_samples)}, val={len(val_samples)}, "
            f"rebalance_freq={site_cfg.rebalance_freq}",
            flush=True,
        )
    if not train_samples or not val_samples:
        raise RuntimeError("No train/validation samples were built.")

    test_features = build_cached_prediction_features(base_config)
    test_signal_index = test_features["mean_ret"].index
    close_df_test = load_close_prices(
        instruments=list(test_features["mean_ret"].columns),
        start_time=pd.Timestamp(test_signal_index.min()) - pd.Timedelta(days=max(180, site_cfg.cov_lookback * 3)),
        end_time=test_signal_index.max(),
    )

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    reports = {}
    metric_rows = {}
    alpha_rows = []

    for eta in site_cfg.etas:
        adapter, result = train_adapter_for_eta(
            eta,
            train_samples,
            val_samples,
            max(len(train_instruments), len(val_instruments)),
            site_cfg,
            core_cfg,
            device,
        )

        test_score = score_from_adapter(test_features, adapter).sort_index()
        markowitz_cfg = MarkowitzConfig(
            top_n=site_cfg.top_n,
            cov_lookback=site_cfg.cov_lookback,
            rebalance_freq=site_cfg.rebalance_freq,
            risk_aversion=eta,
            turnover_penalty=site_cfg.gamma,
            max_weight=site_cfg.max_weight,
            mu_scale=args.mu_scale,
            shrinkage=site_cfg.shrinkage,
            ridge=site_cfg.ridge,
            min_weight=site_cfg.min_weight,
        )
        weight_df = build_markowitz_weight_frame(test_score, close_df_test, markowitz_cfg)
        weight_path = os.path.join(result_dir, f"site_e2e_md_scaled_markowitz_weights_eta_{eta:g}.pkl")
        weight_df.to_pickle(weight_path)

        strategy = RebalanceWeightStrategy(
            signal=signal_frame_to_series(weight_df),
            min_weight=markowitz_cfg.min_weight,
            rebalance_freq=markowitz_cfg.rebalance_freq,
        )
        name = f"SiteE2E-MDScaledMW-eta={eta:g}"
        report_df, metrics = run_single_backtest(strategy, base_config, name)
        reports[name] = report_df
        metric_rows[name] = metrics
        alpha_rows.append({k: v for k, v in result.items() if k != "history"})
        pd.DataFrame(result["history"]).to_csv(
            os.path.join(result_dir, f"site_e2e_md_scaled_train_history_eta_{eta:g}.csv"),
            index=False,
        )

    alpha_df = pd.DataFrame(alpha_rows)
    alpha_path = os.path.join(result_dir, "site_e2e_md_scaled_alpha_summary.csv")
    alpha_df.to_csv(alpha_path, index=False)

    metrics_df = pd.DataFrame(metric_rows).T
    metrics_path = os.path.join(result_dir, "site_e2e_md_scaled_markowitz_metrics.csv")
    metrics_df.to_csv(metrics_path)
    print(metrics_df)

    figure_path = os.path.join(kronos_root, "figures", "Method1_SiteE2E_MeanDownsideScaledMarkowitz.png")
    plot_comparison(reports, base_config.instrument.upper(), figure_path)

    report_path = os.path.join(result_dir, "Method1_SiteE2E_MeanDownsideScaledMarkowitz_Report.MD")
    write_report(report_path, alpha_df, metrics_df, figure_path, split_features, site_cfg, core_cfg, args.mu_scale)
    print(f"Saved scaled Markowitz report to {report_path}")


if __name__ == "__main__":
    main()
