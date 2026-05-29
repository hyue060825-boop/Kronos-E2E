import argparse
import os
import pickle
import sys
from collections import defaultdict
from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import qlib
from qlib.backtest import backtest, executor
from qlib.backtest.decision import TradeDecisionWO
from qlib.config import REG_CN
from qlib.contrib.evaluate import risk_analysis
from qlib.contrib.strategy.signal_strategy import WeightStrategyBase
from qlib.data import D
from qlib.utils.time import Freq

sys.path.append("../")
from config import Config
from model.kronos import auto_regressive_inference
from model_loading import load_predictor, load_tokenizer


CLOSE_IDX = 3


@dataclass
class Method1Config:
    etas: tuple[float, ...] = (0.5, 1.0, 5.0, 10.0)
    top_n: int = 50
    cov_lookback: int = 60
    min_cov_obs: int = 30
    rebalance_freq: int = 5
    gamma: float = 0.02
    max_weight: float = 0.04
    shrinkage: float = 0.2
    ridge: float = 1e-5
    min_weight: float = 1e-4
    alpha_l2: float = 1e-3
    random_candidates: int = 48
    top_alpha_candidates: int = 8


class ReturnPathDataset(Dataset):
    def __init__(self, data: dict[str, pd.DataFrame], config: Config):
        self.data = data
        self.config = config
        self.window_size = config.lookback_window + config.predict_window
        self.symbols = list(data.keys())
        self.indices = []

        print("Building inference indices for Method 1 return paths...")
        for symbol in self.symbols:
            df = data[symbol].reset_index()
            df["minute"] = df["datetime"].dt.minute
            df["hour"] = df["datetime"].dt.hour
            df["weekday"] = df["datetime"].dt.weekday
            df["day"] = df["datetime"].dt.day
            df["month"] = df["datetime"].dt.month
            self.data[symbol] = df

            num_samples = len(df) - self.window_size + 1
            for i in range(max(num_samples, 0)):
                timestamp = df.iloc[i + config.lookback_window - 1]["datetime"]
                self.indices.append((symbol, i, timestamp))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        symbol, start_idx, timestamp = self.indices[idx]
        df = self.data[symbol]

        context_end = start_idx + self.config.lookback_window
        predict_end = context_end + self.config.predict_window

        context_df = df.iloc[start_idx:context_end]
        predict_df = df.iloc[context_end:predict_end]

        raw_x = context_df[self.config.feature_list].values.astype(np.float32)
        x_stamp = context_df[self.config.time_feature_list].values.astype(np.float32)
        y_stamp = predict_df[self.config.time_feature_list].values.astype(np.float32)

        x_mean = np.mean(raw_x, axis=0)
        x_std = np.std(raw_x, axis=0)
        x = (raw_x - x_mean) / (x_std + 1e-5)
        x = np.clip(x, -self.config.clip, self.config.clip)

        return (
            torch.from_numpy(x),
            torch.from_numpy(x_stamp),
            torch.from_numpy(y_stamp),
            symbol,
            timestamp,
            float(raw_x[-1, CLOSE_IDX]),
            float(x_mean[CLOSE_IDX]),
            float(x_std[CLOSE_IDX]),
        )


def collate_return_path_batch(batch):
    x, x_stamp, y_stamp, symbols, timestamps, current_close, close_mean, close_std = zip(*batch)
    return (
        torch.stack(x, dim=0),
        torch.stack(x_stamp, dim=0),
        torch.stack(y_stamp, dim=0),
        list(symbols),
        list(timestamps),
        np.asarray(current_close, dtype=np.float32),
        np.asarray(close_mean, dtype=np.float32),
        np.asarray(close_std, dtype=np.float32),
    )


def load_models(config: Config, device: str):
    print(f"Loading tokenizer and predictor onto {device}...")
    tokenizer = load_tokenizer(config.finetuned_tokenizer_path).to(device).eval()
    model = load_predictor(config.finetuned_predictor_path).to(device).eval()
    return tokenizer, model


def generate_pred_close_path(
    config: Config,
    device: str,
    split_name: str,
    output_path: str,
    models: tuple | None = None,
) -> dict:
    data_path = os.path.join(config.dataset_path, f"{split_name}_data.pkl")
    print(f"Loading {split_name} data from {data_path}...")
    with open(data_path, "rb") as f:
        split_data = pickle.load(f)

    tokenizer, model = models if models is not None else load_models(config, device)
    dataset = ReturnPathDataset(split_data, config)
    batch_size = max(1, config.backtest_batch_size // config.inference_sample_count)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(1, (os.cpu_count() or 2) // 2),
        collate_fn=collate_return_path_batch,
    )

    records = []
    torch_device = torch.device(device)
    with torch.no_grad():
        for x, x_stamp, y_stamp, symbols, timestamps, current_close, close_mean, close_std in tqdm(
            loader,
            desc=f"Generating {split_name} close paths",
        ):
            preds = auto_regressive_inference(
                tokenizer,
                model,
                x.to(torch_device),
                x_stamp.to(torch_device),
                y_stamp.to(torch_device),
                max_context=config.max_context,
                pred_len=config.predict_window,
                clip=config.clip,
                T=config.inference_T,
                top_k=config.inference_top_k,
                top_p=config.inference_top_p,
                sample_count=config.inference_sample_count,
            )
            preds = preds[:, -config.predict_window :, :]
            pred_close = preds[:, :, CLOSE_IDX] * (close_std[:, None] + 1e-5) + close_mean[:, None]

            for i, symbol in enumerate(symbols):
                records.append(
                    {
                        "datetime": pd.Timestamp(timestamps[i]),
                        "instrument": symbol,
                        "current_close": current_close[i],
                        "pred_close": pred_close[i].astype(np.float32),
                    }
                )

    dates = sorted({r["datetime"] for r in records})
    instruments = sorted({r["instrument"] for r in records})
    date_pos = {dt: i for i, dt in enumerate(dates)}
    inst_pos = {inst: i for i, inst in enumerate(instruments)}

    path_arr = np.full((len(dates), len(instruments), config.predict_window), np.nan, dtype=np.float32)
    current_arr = np.full((len(dates), len(instruments)), np.nan, dtype=np.float32)
    for rec in records:
        i = date_pos[rec["datetime"]]
        j = inst_pos[rec["instrument"]]
        path_arr[i, j, :] = rec["pred_close"]
        current_arr[i, j] = rec["current_close"]

    payload = {
        "split": split_name,
        "dates": dates,
        "instruments": instruments,
        "pred_close_path": path_arr,
        "current_close": current_arr,
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"Saved predicted close paths to {output_path}")
    return payload


def pred_close_path_file(config: Config, split_name: str) -> str:
    return os.path.join(
        config.backtest_result_path,
        config.backtest_save_folder_name,
        f"pred_close_path_{split_name}.pkl",
    )


def load_or_generate_pred_close_path(
    config: Config,
    device: str,
    split_name: str,
    force: bool = False,
    models: tuple | None = None,
) -> dict:
    output_path = os.path.join(
        config.backtest_result_path,
        config.backtest_save_folder_name,
        f"pred_close_path_{split_name}.pkl",
    )
    if os.path.exists(output_path) and not force:
        print(f"Loading cached predicted close paths from {output_path}...")
        with open(output_path, "rb") as f:
            return pickle.load(f)

    legacy_test_path = os.path.join(
        config.backtest_result_path,
        config.backtest_save_folder_name,
        "pred_close_path.pkl",
    )
    if split_name == "test" and os.path.exists(legacy_test_path) and not force:
        print(f"Loading legacy cached test close paths from {legacy_test_path}...")
        with open(legacy_test_path, "rb") as f:
            payload = pickle.load(f)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(payload, f)
        print(f"Copied legacy test cache to {output_path}")
        return payload

    return generate_pred_close_path(config, device, split_name, output_path, models=models)


def load_or_generate_split_payloads(config: Config, device: str, force: bool = False) -> dict[str, dict]:
    payloads = {}
    models = None
    for split_name in ("train", "val", "test"):
        output_path = pred_close_path_file(config, split_name)
        needs_inference = force or not os.path.exists(output_path)
        legacy_test_path = os.path.join(
            config.backtest_result_path,
            config.backtest_save_folder_name,
            "pred_close_path.pkl",
        )
        if split_name == "test" and os.path.exists(legacy_test_path) and not force:
            needs_inference = False
        if needs_inference and models is None:
            models = load_models(config, device)
        payloads[split_name] = load_or_generate_pred_close_path(
            config,
            device,
            split_name,
            force=force,
            models=models,
        )
    return payloads


def payload_to_return_features(payload: dict, output_path: str | None = None) -> dict[str, pd.DataFrame]:
    dates = pd.to_datetime(payload["dates"])
    instruments = payload["instruments"]
    pred_path = payload["pred_close_path"].astype(np.float64)
    current_close = payload["current_close"].astype(np.float64)
    denom = current_close[:, :, None]

    with np.errstate(divide="ignore", invalid="ignore"):
        ret_path = pred_path / denom - 1.0

    features = {
        "last_ret": pd.DataFrame(ret_path[:, :, -1], index=dates, columns=instruments),
        "mean_ret": pd.DataFrame(np.nanmean(ret_path, axis=2), index=dates, columns=instruments),
        "max_ret": pd.DataFrame(np.nanmax(ret_path, axis=2), index=dates, columns=instruments),
        "min_ret": pd.DataFrame(np.nanmin(ret_path, axis=2), index=dates, columns=instruments),
    }
    features["downside_risk"] = (-features["min_ret"]).clip(lower=0.0)
    features["path_volatility"] = features["max_ret"] - features["min_ret"]

    if output_path is not None:
        with open(output_path, "wb") as f:
            pickle.dump(features, f)
        print(f"Saved return features to {output_path}")
    return features


def make_alpha_candidates(n_random: int, seed: int = 100) -> list[np.ndarray]:
    base = [
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.3, 0.0],
        [0.7, 0.3, 0.0, 0.3, 0.1],
        [0.6, 0.2, 0.2, 0.4, 0.1],
        [0.5, 0.3, 0.2, 0.6, 0.2],
        [0.4, 0.4, 0.2, 0.8, 0.4],
    ]
    rng = np.random.default_rng(seed)
    candidates = [normalize_alpha(np.asarray(a, dtype=float)) for a in base]
    for _ in range(n_random):
        reward = rng.dirichlet([2.0, 1.5, 1.0])
        penalty_strength = rng.uniform(0.0, 1.2, size=2)
        alpha = np.r_[reward, penalty_strength]
        candidates.append(normalize_alpha(alpha))

    unique = []
    seen = set()
    for alpha in candidates:
        key = tuple(np.round(alpha, 4))
        if key not in seen:
            unique.append(alpha)
            seen.add(key)
    return unique


def normalize_alpha(alpha: np.ndarray) -> np.ndarray:
    alpha = np.clip(alpha.astype(float), 0.0, None)
    denom = np.sum(np.abs(alpha))
    if denom <= 0:
        return np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    return alpha / denom


def build_score(features: dict[str, pd.DataFrame], alpha: np.ndarray) -> pd.DataFrame:
    return (
        alpha[0] * features["mean_ret"]
        + alpha[1] * features["last_ret"]
        + alpha[2] * features["max_ret"]
        - alpha[3] * features["downside_risk"]
        - alpha[4] * features["path_volatility"]
    )


def load_close_prices(instruments: list[str], start_time, end_time) -> pd.DataFrame:
    raw = D.features(instruments, ["$close"], start_time=start_time, end_time=end_time, freq="day")
    close = raw["$close"].unstack(level="instrument").sort_index()
    close.index = pd.to_datetime(close.index)
    return close


def estimate_covariance(history_returns: pd.DataFrame, cfg: Method1Config) -> pd.DataFrame:
    history_returns = history_returns.dropna(axis=1, thresh=cfg.min_cov_obs)
    if history_returns.empty:
        return pd.DataFrame()

    cov = history_returns.cov()
    diag = np.diag(np.diag(cov.values))
    shrunk = (1.0 - cfg.shrinkage) * cov.values + cfg.shrinkage * diag
    shrunk = np.nan_to_num(shrunk, nan=0.0, posinf=0.0, neginf=0.0)
    shrunk = (shrunk + shrunk.T) / 2.0
    shrunk += np.eye(shrunk.shape[0]) * cfg.ridge
    return pd.DataFrame(shrunk, index=cov.index, columns=cov.columns)


def solve_markowitz_weights(mu: pd.Series, cov: pd.DataFrame, prev_weights: pd.Series, eta: float, cfg: Method1Config) -> pd.Series:
    assets = cov.index.intersection(mu.dropna().index)
    if len(assets) == 0:
        return pd.Series(dtype=float)

    mu = mu.reindex(assets).fillna(0.0).astype(float)
    cov = cov.reindex(index=assets, columns=assets).fillna(0.0)
    prev = prev_weights.reindex(assets).fillna(0.0).astype(float)
    n_assets = len(assets)
    effective_max_weight = max(cfg.max_weight, 1.0 / n_assets + 1e-6)

    w = cp.Variable(n_assets)
    objective = cp.Maximize(
        mu.values @ w
        - 0.5 * eta * cp.quad_form(w, cp.psd_wrap(cov.values))
        - 0.5 * cfg.gamma * cp.sum_squares(w - prev.values)
    )
    constraints = [w >= 0, cp.sum(w) == 1, w <= effective_max_weight]
    problem = cp.Problem(objective, constraints)

    for solver in ("OSQP", "CLARABEL", "SCS"):
        try:
            problem.solve(solver=solver, warm_start=True, verbose=False)
        except Exception:
            continue
        if w.value is not None and problem.status in {"optimal", "optimal_inaccurate"}:
            weights = np.asarray(w.value, dtype=float)
            weights = np.clip(weights, 0.0, None)
            if weights.sum() > 0:
                weights = weights / weights.sum()
                result = pd.Series(weights, index=assets)
                return result[result > cfg.min_weight]

    fallback_assets = mu.sort_values(ascending=False).head(max(1, min(n_assets, cfg.top_n))).index
    return pd.Series(1.0 / len(fallback_assets), index=fallback_assets)


def build_weight_frame(score_df: pd.DataFrame, close_df: pd.DataFrame, eta: float, cfg: Method1Config) -> pd.DataFrame:
    returns = close_df.ffill().pct_change(fill_method=None)
    weights = pd.DataFrame(0.0, index=score_df.index, columns=score_df.columns)
    prev_weights = pd.Series(0.0, index=score_df.columns)

    for step, dt in enumerate(score_df.index):
        if step > 0 and step % cfg.rebalance_freq != 0:
            weights.loc[dt] = prev_weights.reindex(weights.columns).fillna(0.0)
            continue

        score = score_df.loc[dt].dropna()
        if score.empty:
            continue

        top_assets = score.sort_values(ascending=False).head(cfg.top_n).index
        held_assets = prev_weights[prev_weights > cfg.min_weight].index
        candidate_assets = pd.Index(top_assets).union(held_assets).intersection(close_df.columns)
        history = returns.loc[returns.index < dt, candidate_assets].tail(cfg.cov_lookback)
        cov = estimate_covariance(history, cfg)

        if cov.empty:
            selected = score.reindex(candidate_assets).dropna().sort_values(ascending=False).head(cfg.top_n)
            day_weights = pd.Series(1.0 / len(selected), index=selected.index) if len(selected) else pd.Series(dtype=float)
        else:
            day_weights = solve_markowitz_weights(score, cov, prev_weights, eta, cfg)

        if not day_weights.empty:
            weights.loc[dt, day_weights.index] = day_weights.values
            prev_weights = weights.loc[dt]

    return weights


def fast_realized_utility(weights: pd.DataFrame, close_df: pd.DataFrame, eta: float, cfg: Method1Config, dates: pd.Index) -> float:
    returns = close_df.ffill().pct_change(fill_method=None)
    next_returns = returns.shift(-1)
    values = []
    prev_w = pd.Series(0.0, index=weights.columns)

    for dt in dates:
        if dt not in weights.index or dt not in next_returns.index:
            continue
        w = weights.loc[dt].fillna(0.0)
        r_next = next_returns.loc[dt].reindex(weights.columns).fillna(0.0)
        hist = returns.loc[returns.index < dt, weights.columns].tail(cfg.cov_lookback)
        cov = hist.cov().reindex(index=weights.columns, columns=weights.columns).fillna(0.0)

        realized_ret = float((w * r_next).sum())
        risk = float(w.values @ cov.values @ w.values) if len(cov) else 0.0
        turnover = float(np.abs(w - prev_w).sum())
        alpha_penalty = 0.0
        values.append(realized_ret - 0.5 * eta * risk - cfg.gamma * turnover - alpha_penalty)
        prev_w = w

    return float(np.nanmean(values)) if values else -np.inf


def feature_dates(features: dict[str, pd.DataFrame]) -> pd.Index:
    return pd.Index(pd.to_datetime(features["mean_ret"].index)).sort_values()


def train_alpha_for_eta(
    eta: float,
    train_features: dict[str, pd.DataFrame],
    train_close_df: pd.DataFrame,
    val_features: dict[str, pd.DataFrame],
    val_close_df: pd.DataFrame,
    candidates: list[np.ndarray],
    cfg: Method1Config,
) -> tuple[np.ndarray, float, float]:
    best_alpha = candidates[0]
    best_val_utility = -np.inf
    best_train_utility = -np.inf
    train_rows = []
    train_dates = feature_dates(train_features)
    val_dates = feature_dates(val_features)

    for alpha in tqdm(candidates, desc=f"Training alpha eta={eta:g}", leave=False):
        train_score = build_score(train_features, alpha)
        train_weights = build_weight_frame(train_score, train_close_df, eta, cfg)
        train_utility = fast_realized_utility(train_weights, train_close_df, eta, cfg, train_dates)
        train_utility -= cfg.alpha_l2 * float(np.sum(alpha * alpha))
        train_rows.append((train_utility, alpha))

    train_rows.sort(key=lambda item: item[0], reverse=True)
    top_rows = train_rows[: max(1, cfg.top_alpha_candidates)]

    for train_utility, alpha in tqdm(top_rows, desc=f"Validating alpha eta={eta:g}", leave=False):
        val_score = build_score(val_features, alpha)
        val_weights = build_weight_frame(val_score, val_close_df, eta, cfg)
        val_utility = fast_realized_utility(val_weights, val_close_df, eta, cfg, val_dates)
        val_utility -= cfg.alpha_l2 * float(np.sum(alpha * alpha))
        if val_utility > best_val_utility:
            best_val_utility = val_utility
            best_train_utility = train_utility
            best_alpha = alpha

    return best_alpha, best_train_utility, best_val_utility


def signal_frame_to_series(signal_df: pd.DataFrame) -> pd.Series:
    signal_series = signal_df.stack()
    signal_series.index.names = ["datetime", "instrument"]
    return signal_series.swaplevel().sort_index()


class RebalanceWeightStrategy(WeightStrategyBase):
    def __init__(self, *, min_weight: float = 1e-4, rebalance_freq: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.min_weight = min_weight
        self.rebalance_freq = rebalance_freq

    def generate_trade_decision(self, execute_result=None):
        trade_step = self.trade_calendar.get_trade_step()
        if trade_step > 0 and trade_step % self.rebalance_freq != 0:
            return TradeDecisionWO([], self)
        return super().generate_trade_decision(execute_result=execute_result)

    def generate_target_weight_position(self, score, current, trade_start_time, trade_end_time):
        if isinstance(score, pd.DataFrame):
            score = score.iloc[:, 0]
        if score is None or len(score) == 0:
            return {}
        weights = score.dropna().astype(float)
        weights = weights[weights > self.min_weight]
        if weights.empty:
            return {}
        weights = weights / weights.sum()
        return weights.to_dict()


def get_analysis_value(analysis: pd.DataFrame, field: str) -> float:
    if field in analysis.index:
        value = analysis.loc[field]
        if isinstance(value, pd.Series):
            return float(value.iloc[0])
        return float(value)
    if field in analysis.columns:
        value = analysis[field]
        if isinstance(value, pd.Series):
            return float(value.iloc[0])
        return float(value)
    return float("nan")


def run_single_backtest(weight_df: pd.DataFrame, config: Config, cfg: Method1Config, name: str):
    strategy = RebalanceWeightStrategy(
        signal=signal_frame_to_series(weight_df),
        min_weight=cfg.min_weight,
        rebalance_freq=cfg.rebalance_freq,
    )
    executor_config = {
        "time_per_step": "day",
        "generate_portfolio_metrics": True,
        "delay_execution": True,
    }
    backtest_config = {
        "start_time": config.backtest_time_range[0],
        "end_time": config.backtest_time_range[1],
        "account": 100_000_000,
        "benchmark": config.backtest_benchmark,
        "exchange_kwargs": {
            "freq": "day",
            "limit_threshold": 0.095,
            "deal_price": "open",
            "open_cost": 0.001,
            "close_cost": 0.0015,
            "min_cost": 5,
        },
        "executor": executor.SimulatorExecutor(**executor_config),
    }

    print(f"\nBacktesting {name}...")
    portfolio_metric_dict, _ = backtest(strategy=strategy, **backtest_config)
    analysis_freq = "{0}{1}".format(*Freq.parse("day"))
    report, _ = portfolio_metric_dict.get(analysis_freq)
    excess_with_cost = report["return"] - report["bench"] - report["cost"]
    total_with_cost = report["return"] - report["cost"]
    analysis = risk_analysis(excess_with_cost, freq=analysis_freq)
    total_analysis = risk_analysis(total_with_cost, freq=analysis_freq)
    metrics = {
        "annualized_return_w_cost": get_analysis_value(total_analysis, "annualized_return"),
        "annualized_excess_return_w_cost": get_analysis_value(analysis, "annualized_return"),
        "information_ratio": get_analysis_value(analysis, "information_ratio"),
        "max_drawdown": get_analysis_value(analysis, "max_drawdown"),
    }
    print(pd.Series(metrics, name=name))
    report_df = pd.DataFrame(
        {
            "cum_bench": report["bench"].cumsum(),
            "cum_return_w_cost": total_with_cost.cumsum(),
            "cum_ex_return_w_cost": excess_with_cost.cumsum(),
        }
    )
    return report_df, metrics


def plot_eta_results(reports: dict[str, pd.DataFrame], benchmark_name: str, save_path: str) -> None:
    return_df = pd.DataFrame({name: df["cum_return_w_cost"] for name, df in reports.items()})
    ex_return_df = pd.DataFrame({name: df["cum_ex_return_w_cost"] for name, df in reports.items()})
    benchmark = next(iter(reports.values()))["cum_bench"]

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    return_df.plot(ax=axes[0], grid=True, linewidth=1.8)
    axes[0].plot(benchmark, label=benchmark_name, color="black", linestyle="--", linewidth=1.4)
    axes[0].set_title("Method 1 Return-Aware Adapter: Cumulative Return with Cost")
    axes[0].set_ylabel("Cumulative Return")
    axes[0].legend()

    ex_return_df.plot(ax=axes[1], grid=True, linewidth=1.8)
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_title("Cumulative Excess Return with Cost")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Cumulative Excess Return")
    axes[1].legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Saved Method 1 figure to {save_path}")


def write_report(
    save_path: str,
    alpha_rows: list[dict],
    metrics_df: pd.DataFrame,
    figure_path: str,
    cfg: Method1Config,
    split_features: dict[str, dict[str, pd.DataFrame]],
    config: Config,
) -> None:
    alpha_df = pd.DataFrame(alpha_rows)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("# Method 1 Return-Aware Adapter 结果对比\n\n")
        f.write("本实验冻结 Kronos，不更新 Tokenizer / Predictor，仅把 Kronos 未来 10 步 close 预测路径转化为收益率型 score，并在不同风险厌恶系数下训练不同的 alpha。\n\n")
        f.write("## 数据切分\n\n")
        f.write("Method 1 使用与原 Kronos 微调/回测链路一致的 processed dataset，而不是把 2020 测试信号再切成训练集。\n\n")
        for split_name in ("train", "val", "test"):
            dates = feature_dates(split_features[split_name])
            f.write(
                f"- `{split_name}_data.pkl`: {dates.min().date()} 至 {dates.max().date()}，"
                f"共 {len(dates)} 个信号日\n"
            )
        f.write(
            f"- 最终 Qlib 回测区间: {config.backtest_time_range[0]} 至 {config.backtest_time_range[1]}\n\n"
        )
        f.write("## 训练得到的 Alpha\n\n")
        f.write(dataframe_to_markdown(alpha_df, index=False, floatfmt=".6f"))
        f.write("\n\n")
        f.write("alpha 含义：\n\n")
        f.write("- `alpha_mean`: 平均预测收益权重\n")
        f.write("- `alpha_last`: 第 10 步预测收益权重\n")
        f.write("- `alpha_max`: 最大上行空间权重\n")
        f.write("- `alpha_downside`: 下行风险惩罚权重\n")
        f.write("- `alpha_volatility`: 路径波动惩罚权重\n\n")
        f.write("## 回测结果\n\n")
        f.write(dataframe_to_markdown(metrics_df, index=True, floatfmt=".6f"))
        f.write("\n\n")
        f.write("## 结果图\n\n")
        f.write(f"![Method 1 Result]({os.path.relpath(figure_path, os.path.dirname(save_path))})\n\n")
        f.write("## 参数\n\n")
        f.write(f"- top_n: {cfg.top_n}\n")
        f.write(f"- cov_lookback: {cfg.cov_lookback}\n")
        f.write(f"- rebalance_freq: {cfg.rebalance_freq}\n")
        f.write(f"- gamma: {cfg.gamma}\n")
        f.write(f"- max_weight: {cfg.max_weight}\n")
        f.write(f"- random_candidates: {cfg.random_candidates}\n")
        f.write(f"- top_alpha_candidates: {cfg.top_alpha_candidates}\n")
    print(f"Saved Method 1 report to {save_path}")


def dataframe_to_markdown(df: pd.DataFrame, index: bool, floatfmt: str = ".6f") -> str:
    table = df.copy()
    if index:
        table = table.reset_index().rename(columns={"index": "strategy"})

    def fmt_value(value):
        if isinstance(value, (float, np.floating)):
            return format(float(value), floatfmt)
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        return str(value)

    headers = [str(col) for col in table.columns]
    rows = [[fmt_value(value) for value in row] for row in table.to_numpy()]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def parse_etas(raw: str) -> tuple[float, ...]:
    return tuple(float(x.strip()) for x in raw.split(",") if x.strip())


def main():
    parser = argparse.ArgumentParser(description="Method 1: Frozen-Kronos Return-Aware Preference Adapter")
    parser.add_argument("--device", default="cuda:0", help="Inference device used when pred_close_path.pkl is absent.")
    parser.add_argument("--force_inference", action="store_true", help="Regenerate pred_close_path.pkl even if cached.")
    parser.add_argument("--etas", default="0.5,1.0,5.0,10.0", help="Comma-separated risk aversion coefficients.")
    parser.add_argument("--top_n", type=int, default=50)
    parser.add_argument("--cov_lookback", type=int, default=60)
    parser.add_argument("--rebalance_freq", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=0.02)
    parser.add_argument("--max_weight", type=float, default=0.04)
    parser.add_argument("--random_candidates", type=int, default=48)
    parser.add_argument("--top_alpha_candidates", type=int, default=8)
    args = parser.parse_args()

    kronos_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(kronos_root)

    base_config = Config()
    method_cfg = Method1Config(
        etas=parse_etas(args.etas),
        top_n=args.top_n,
        cov_lookback=args.cov_lookback,
        rebalance_freq=args.rebalance_freq,
        gamma=args.gamma,
        max_weight=args.max_weight,
        random_candidates=args.random_candidates,
        top_alpha_candidates=args.top_alpha_candidates,
    )

    qlib.init(provider_uri=base_config.qlib_data_path, region=REG_CN)

    payloads = load_or_generate_split_payloads(base_config, args.device, force=args.force_inference)
    split_features = {}
    close_dfs = {}
    for split_name, payload in payloads.items():
        feature_path = os.path.join(
            base_config.backtest_result_path,
            base_config.backtest_save_folder_name,
            f"method1_return_features_{split_name}.pkl",
        )
        split_features[split_name] = payload_to_return_features(payload, feature_path)
        score_index = split_features[split_name]["mean_ret"].index
        instruments = list(split_features[split_name]["mean_ret"].columns)
        history_start = pd.Timestamp(score_index.min()) - pd.Timedelta(days=max(180, args.cov_lookback * 3))
        close_dfs[split_name] = load_close_prices(
            instruments,
            start_time=history_start,
            end_time=score_index.max(),
        )

    candidates = make_alpha_candidates(method_cfg.random_candidates, seed=base_config.seed)

    reports = {}
    metric_rows = {}
    alpha_rows = []

    for eta in method_cfg.etas:
        alpha, train_utility, val_utility = train_alpha_for_eta(
            eta,
            split_features["train"],
            close_dfs["train"],
            split_features["val"],
            close_dfs["val"],
            candidates,
            method_cfg,
        )
        test_score = build_score(split_features["test"], alpha)
        weights = build_weight_frame(test_score, close_dfs["test"], eta, method_cfg)

        weight_path = os.path.join(
            base_config.backtest_result_path,
            base_config.backtest_save_folder_name,
            f"method1_weights_eta_{eta:g}.pkl",
        )
        weights.to_pickle(weight_path)

        name = f"Method1-eta={eta:g}"
        report_df, metrics = run_single_backtest(weights, base_config, method_cfg, name)
        reports[name] = report_df
        metric_rows[name] = metrics
        alpha_rows.append(
            {
                "eta": eta,
                "alpha_mean": alpha[0],
                "alpha_last": alpha[1],
                "alpha_max": alpha[2],
                "alpha_downside": alpha[3],
                "alpha_volatility": alpha[4],
                "train_utility": train_utility,
                "validation_utility": val_utility,
            }
        )

    metrics_df = pd.DataFrame(metric_rows).T
    metrics_path = os.path.join(
        base_config.backtest_result_path,
        base_config.backtest_save_folder_name,
        "method1_return_adapter_metrics.csv",
    )
    metrics_df.to_csv(metrics_path)
    print(f"Saved metrics to {metrics_path}")
    print(metrics_df)

    figure_path = os.path.join(kronos_root, "figures", "Method1_Return_Aware_Adapter.png")
    plot_eta_results(reports, base_config.instrument.upper(), figure_path)

    report_path = os.path.join(
        base_config.backtest_result_path,
        base_config.backtest_save_folder_name,
        "Method1_Return_Aware_Adapter_Report.MD",
    )
    write_report(report_path, alpha_rows, metrics_df, figure_path, method_cfg, split_features, base_config)


if __name__ == "__main__":
    main()
