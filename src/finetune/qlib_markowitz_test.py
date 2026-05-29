import argparse
import os
import pickle
import sys
from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import qlib
from qlib.backtest import backtest, executor
from qlib.config import REG_CN
from qlib.backtest.decision import TradeDecisionWO
from qlib.contrib.evaluate import risk_analysis
from qlib.contrib.strategy import TopkDropoutStrategy
from qlib.contrib.strategy.signal_strategy import WeightStrategyBase
from qlib.data import D
from qlib.utils.time import Freq

sys.path.append("../")
from config import Config


def cs_zscore(df: pd.DataFrame) -> pd.DataFrame:
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1) + 1e-8, axis=0)


def load_cached_predictions(config: Config) -> dict[str, pd.DataFrame]:
    predictions_file = os.path.join(
        config.backtest_result_path,
        config.backtest_save_folder_name,
        "predictions.pkl",
    )
    if not os.path.exists(predictions_file):
        raise FileNotFoundError(
            f"Cached predictions not found: {predictions_file}. "
            "Please run finetune/qlib_test.py once to generate predictions.pkl."
        )

    print(f"Loading cached prediction signals from {predictions_file}...")
    with open(predictions_file, "rb") as f:
        return pickle.load(f)


def add_markowitz_candidate_signals(prediction_dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    prediction_dfs = prediction_dfs.copy()
    required = {"last", "mean", "max", "min"}
    missing = required - set(prediction_dfs)
    if missing:
        raise KeyError(f"Missing base prediction signals: {', '.join(sorted(missing))}")

    mean_z = cs_zscore(prediction_dfs["mean"])
    min_z = cs_zscore(prediction_dfs["min"])
    downside_z = cs_zscore((-prediction_dfs["min"]).clip(lower=0))
    volatility_z = cs_zscore((prediction_dfs["max"] - prediction_dfs["min"]).abs())
    slope_proxy_z = cs_zscore(prediction_dfs["last"] - prediction_dfs["mean"])
    # mean_z = prediction_dfs["mean"]
    # min_z = prediction_dfs["min"]
    # downside_z = (-prediction_dfs["min"]).clip(lower=0)
    # volatility_z = (prediction_dfs["max"] - prediction_dfs["min"]).abs()
    # slope_proxy_z = prediction_dfs["last"] - prediction_dfs["mean"]

    prediction_dfs["mean_ret"] = mean_z
    prediction_dfs["q25_ret"] = 0.75 * min_z + 0.25 * mean_z
    prediction_dfs["mean_downside_ret"] = mean_z - 0.3 * downside_z
    prediction_dfs["mean_q25_ret"] = 0.7 * mean_z + 0.3 * prediction_dfs["q25_ret"]
    prediction_dfs["mean_stability_ret"] = mean_z - 0.2 * volatility_z
    prediction_dfs["slope_mean_ret"] = 0.7 * mean_z + 0.3 * slope_proxy_z
    return prediction_dfs


def signal_frame_to_series(signal_df: pd.DataFrame) -> pd.Series:
    signal_series = signal_df.stack()
    signal_series.index.names = ["datetime", "instrument"]
    return signal_series.swaplevel().sort_index()


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


class RebalanceWeightStrategy(WeightStrategyBase):
    """Qlib strategy that rebalances to precomputed target weights."""

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


@dataclass
class MarkowitzConfig:
    signal_name: str = "mean_downside_ret"
    top_n: int = 50
    cov_lookback: int = 60
    min_cov_obs: int = 30
    rebalance_freq: int = 5
    risk_aversion: float = 2.5
    turnover_penalty: float = 0.02
    max_weight: float = 0.04
    mu_scale: float = 0.003
    shrinkage: float = 0.2
    ridge: float = 1e-5
    min_weight: float = 1e-4


def load_close_prices(instruments: list[str], start_time, end_time) -> pd.DataFrame:
    raw = D.features(instruments, ["$close"], start_time=start_time, end_time=end_time, freq="day")
    close = raw["$close"].unstack(level="instrument").sort_index()
    close.index = pd.to_datetime(close.index)
    return close


def estimate_covariance(history_returns: pd.DataFrame, cfg: MarkowitzConfig) -> pd.DataFrame:
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


def solve_markowitz_weights(
    mu: pd.Series,
    cov: pd.DataFrame,
    prev_weights: pd.Series,
    cfg: MarkowitzConfig,
) -> pd.Series:
    assets = cov.index.intersection(mu.dropna().index)
    if len(assets) == 0:
        return pd.Series(dtype=float)

    mu = mu.reindex(assets).fillna(0.0).astype(float) * cfg.mu_scale
    cov = cov.reindex(index=assets, columns=assets).fillna(0.0)
    prev = prev_weights.reindex(assets).fillna(0.0).astype(float)

    n_assets = len(assets)
    effective_max_weight = max(cfg.max_weight, 1.0 / n_assets + 1e-6)

    w = cp.Variable(n_assets)
    objective = cp.Maximize(
        mu.values @ w
        - cfg.risk_aversion * cp.quad_form(w, cp.psd_wrap(cov.values))
        - cfg.turnover_penalty * cp.norm1(w - prev.values)
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

    # Robust fallback: equal weight the strongest names if the QP solver fails.
    fallback_assets = mu.sort_values(ascending=False).head(max(1, min(n_assets, cfg.top_n))).index
    return pd.Series(1.0 / len(fallback_assets), index=fallback_assets)


def build_markowitz_weight_frame(
    signal_df: pd.DataFrame,
    close_df: pd.DataFrame,
    cfg: MarkowitzConfig,
) -> pd.DataFrame:
    returns = close_df.ffill().pct_change(fill_method=None)
    weights = pd.DataFrame(0.0, index=signal_df.index, columns=signal_df.columns)
    prev_weights = pd.Series(0.0, index=signal_df.columns)

    for step, dt in enumerate(signal_df.index):
        if step > 0 and step % cfg.rebalance_freq != 0:
            weights.loc[dt] = prev_weights.reindex(weights.columns).fillna(0.0)
            continue

        score = signal_df.loc[dt].dropna()
        if score.empty:
            continue

        top_assets = score.sort_values(ascending=False).head(cfg.top_n).index
        held_assets = prev_weights[prev_weights > cfg.min_weight].index
        candidate_assets = pd.Index(top_assets).union(held_assets).intersection(close_df.columns)

        history = returns.loc[returns.index < dt, candidate_assets].tail(cfg.cov_lookback)
        cov = estimate_covariance(history, cfg)
        if cov.empty:
            selected = score.reindex(candidate_assets).dropna().sort_values(ascending=False).head(cfg.top_n)
            if not selected.empty:
                day_weights = pd.Series(1.0 / len(selected), index=selected.index)
            else:
                day_weights = pd.Series(dtype=float)
        else:
            day_weights = solve_markowitz_weights(score, cov, prev_weights, cfg)

        if not day_weights.empty:
            weights.loc[dt, day_weights.index] = day_weights.values
            prev_weights = weights.loc[dt]

    return weights


def build_equal_topn_weight_frame(
    signal_df: pd.DataFrame,
    top_n: int,
    rebalance_freq: int,
) -> pd.DataFrame:
    weights = pd.DataFrame(0.0, index=signal_df.index, columns=signal_df.columns)
    prev_weights = pd.Series(0.0, index=signal_df.columns)

    for step, dt in enumerate(signal_df.index):
        if step > 0 and step % rebalance_freq != 0:
            weights.loc[dt] = prev_weights.reindex(weights.columns).fillna(0.0)
            continue

        score = signal_df.loc[dt].dropna()
        if score.empty:
            continue

        top_assets = score.sort_values(ascending=False).head(top_n).index
        if len(top_assets) == 0:
            continue

        day_weights = pd.Series(0.0, index=signal_df.columns)
        day_weights.loc[top_assets] = 1.0 / len(top_assets)
        weights.loc[dt] = day_weights
        prev_weights = day_weights

    return weights


def run_single_backtest(strategy, config: Config, name: str) -> tuple[pd.DataFrame, dict[str, float]]:
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
    analysis = risk_analysis(excess_with_cost, freq=analysis_freq)
    metrics = {
        "annualized_excess_return": get_analysis_value(analysis, "annualized_return"),
        "information_ratio": get_analysis_value(analysis, "information_ratio"),
        "max_drawdown": get_analysis_value(analysis, "max_drawdown"),
    }
    print(pd.Series(metrics, name=name))

    report_df = pd.DataFrame(
        {
            "cum_bench": report["bench"].cumsum(),
            "cum_return_w_cost": (report["return"] - report["cost"]).cumsum(),
            "cum_ex_return_w_cost": excess_with_cost.cumsum(),
        }
    )
    return report_df, metrics


def plot_comparison(
    reports: dict[str, pd.DataFrame],
    benchmark_name: str,
    save_path: str,
) -> None:
    return_df = pd.DataFrame({name: df["cum_return_w_cost"] for name, df in reports.items()})
    ex_return_df = pd.DataFrame({name: df["cum_ex_return_w_cost"] for name, df in reports.items()})
    benchmark = next(iter(reports.values()))["cum_bench"]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    return_df.plot(ax=axes[0], grid=True, linewidth=1.8)
    axes[0].plot(benchmark, label=benchmark_name, color="black", linestyle="--", linewidth=1.5)
    axes[0].set_title("Kronos TopK / EqualTopN / Markowitz: Cumulative Return with Cost")
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
    print(f"\nSaved comparison figure to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Run Kronos-Markowitz backtest against Kronos TopK.")
    parser.add_argument("--signal", default="mean_downside_ret", help="Kronos signal used as expected return proxy.")
    parser.add_argument("--top_n", type=int, default=50, help="Candidate asset count for Markowitz optimizer.")
    parser.add_argument("--cov_lookback", type=int, default=60, help="Lookback days for covariance estimation.")
    parser.add_argument("--rebalance_freq", type=int, default=5, help="Rebalance every N trading days.")
    parser.add_argument("--eta", type=float, default=2.5, help="Risk aversion coefficient.")
    parser.add_argument("--gamma", type=float, default=0.02, help="Turnover penalty coefficient.")
    parser.add_argument("--w_max", type=float, default=0.04, help="Single-stock max weight.")
    parser.add_argument("--mu_scale", type=float, default=0.003, help="Scale z-scored Kronos signal into return proxy.")
    args = parser.parse_args()

    kronos_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(kronos_root)

    base_config = Config()
    qlib.init(provider_uri=base_config.qlib_data_path, region=REG_CN)

    prediction_dfs = add_markowitz_candidate_signals(load_cached_predictions(base_config))
    if args.signal not in prediction_dfs:
        raise KeyError(f"Signal {args.signal!r} not found. Available signals: {sorted(prediction_dfs)}")

    signal_df = prediction_dfs[args.signal].sort_index()
    markowitz_cfg = MarkowitzConfig(
        signal_name=args.signal,
        top_n=args.top_n,
        cov_lookback=args.cov_lookback,
        rebalance_freq=args.rebalance_freq,
        risk_aversion=args.eta,
        turnover_penalty=args.gamma,
        max_weight=args.w_max,
        mu_scale=args.mu_scale,
    )

    history_start = pd.Timestamp(signal_df.index.min()) - pd.Timedelta(days=max(180, args.cov_lookback * 3))
    close_df = load_close_prices(
        instruments=list(signal_df.columns),
        start_time=history_start,
        end_time=signal_df.index.max(),
    )

    print("\nBuilding Kronos-Markowitz target weights...")
    weight_df = build_markowitz_weight_frame(signal_df, close_df, markowitz_cfg)
    weight_path = os.path.join(
        base_config.backtest_result_path,
        base_config.backtest_save_folder_name,
        "Kronos_Markowitz_weights.pkl",
    )
    os.makedirs(os.path.dirname(weight_path), exist_ok=True)
    weight_df.to_pickle(weight_path)
    print(f"Saved Markowitz weights to {weight_path}")

    topk_strategy = TopkDropoutStrategy(
        topk=base_config.backtest_n_symbol_hold,
        n_drop=base_config.backtest_n_symbol_drop,
        hold_thresh=base_config.backtest_hold_thresh,
        signal=signal_frame_to_series(signal_df),
    )
    equal_weight_df = build_equal_topn_weight_frame(
        signal_df=signal_df,
        top_n=markowitz_cfg.top_n,
        rebalance_freq=markowitz_cfg.rebalance_freq,
    )
    markowitz_strategy = RebalanceWeightStrategy(
        signal=signal_frame_to_series(weight_df),
        min_weight=markowitz_cfg.min_weight,
        rebalance_freq=markowitz_cfg.rebalance_freq,
    )

    topk_report, topk_metrics = run_single_backtest(topk_strategy, base_config, f"Kronos-TopK-{args.signal}")
    equal_report, equal_metrics = run_single_backtest(
        RebalanceWeightStrategy(
            signal=signal_frame_to_series(equal_weight_df),
            min_weight=markowitz_cfg.min_weight,
            rebalance_freq=markowitz_cfg.rebalance_freq,
        ),
        base_config,
        f"Kronos-EqualTopN-{args.signal}",
    )
    markowitz_report, markowitz_metrics = run_single_backtest(
        markowitz_strategy,
        base_config,
        f"Kronos-Markowitz-{args.signal}",
    )

    metrics_df = pd.DataFrame(
        {
            f"Kronos-TopK-{args.signal}": topk_metrics,
            f"Kronos-EqualTopN-{args.signal}": equal_metrics,
            f"Kronos-Markowitz-{args.signal}": markowitz_metrics,
        }
    ).T
    metrics_path = os.path.join(
        base_config.backtest_result_path,
        base_config.backtest_save_folder_name,
        "Kronos_Markowitz_metrics.csv",
    )
    metrics_df.to_csv(metrics_path)
    print(f"\nSaved metrics to {metrics_path}")
    print(metrics_df)

    figure_path = os.path.join(kronos_root, "figures", "Kronos_Markowitz_result.png")
    plot_comparison(
        reports={
            f"Kronos-TopK-{args.signal}": topk_report,
            f"Kronos-EqualTopN-{args.signal}": equal_report,
            f"Kronos-Markowitz-{args.signal}": markowitz_report,
        },
        benchmark_name=base_config.instrument.upper(),
        save_path=os.path.join(kronos_root, "figures", "Kronos_Markowitz_result1.png"),
    )


if __name__ == "__main__":
    main()
