"""Evaluation metrics for time-series synthetic data.

Provides quality comparison between real and synthetic time-series DataFrames
using statistical tests and autocorrelation similarity.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _autocorrelation(series: pd.Series, lag: int = 1) -> float:  # type: ignore[type-arg]
    """Calculate lag-N autocorrelation for a series."""
    if len(series) <= lag:
        return 0.0
    val = series.autocorr(lag=lag)
    return 0.0 if np.isnan(val) else float(val)


def _sequence_autocorrelation(df: pd.DataFrame, sequence_key: str, col: str, lag: int = 1) -> float:
    """Calculate average autocorrelation across all sequences for a column."""
    autocorrs = []
    for _, seq_df in df.groupby(sequence_key):
        val = _autocorrelation(seq_df[col], lag=lag)
        autocorrs.append(val)
    return float(np.mean(autocorrs)) if autocorrs else 0.0


def _mean_std_similarity(
    real_col: pd.Series,
    synthetic_col: pd.Series,  # type: ignore[type-arg]
) -> float:
    """Compare mean and standard deviation similarity (0 to 1)."""
    r_mean = real_col.mean()
    s_mean = synthetic_col.mean()
    r_std = real_col.std()
    s_std = synthetic_col.std()

    mean_diff = 0.0
    if r_mean == 0 and s_mean == 0:
        pass
    elif r_mean == 0:
        mean_diff = 1.0
    else:
        mean_diff = min(abs(r_mean - s_mean) / (abs(r_mean) + 1e-10), 1.0)

    std_diff = 0.0
    if r_std == 0 and s_std == 0:
        pass
    elif r_std == 0 or np.isnan(r_std):
        std_diff = 1.0 if not np.isnan(s_std) and s_std != 0 else 0.0
    else:
        std_diff = min(abs(r_std - s_std) / (abs(r_std) + 1e-10), 1.0)

    return float(1.0 - np.mean([mean_diff, std_diff]))


def evaluate_timeseries(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    sequence_key: str,
    metadata: Any = None,
) -> dict[str, Any]:
    """Evaluate quality of synthetic time-series data against original.

    Parameters
    ----------
    real_data : pd.DataFrame
        Original sample data.
    synthetic_data : pd.DataFrame
        Generated synthetic data.
    sequence_key : str
        Column used to group data into sequences.
    metadata : Any, optional
        SDV metadata object. If provided and SDV is installed, runs
        SDV's built-in quality evaluation alongside custom metrics.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys:
        - ``mean_std_similarity``: dict of per-column distribution similarity scores
        - ``autocorr_similarity``: dict of per-column autocorrelation similarity scores
        - ``overall_score``: weighted average of all metrics
        - ``sdv_quality`` (optional): SDV's built-in quality report if available
    """
    if not isinstance(real_data, pd.DataFrame):
        raise TypeError(f"real_data must be a DataFrame, got {type(real_data).__name__}")
    if not isinstance(synthetic_data, pd.DataFrame):
        raise TypeError(f"synthetic_data must be a DataFrame, got {type(synthetic_data).__name__}")

    if sequence_key not in real_data.columns or sequence_key not in synthetic_data.columns:
        raise ValueError(f"sequence_key '{sequence_key}' missing from data.")

    numeric_cols = real_data.select_dtypes(include=np.number).columns.tolist()
    common_cols = [c for c in numeric_cols if c in synthetic_data.columns and c != sequence_key]

    if not common_cols:
        raise ValueError("No common numeric columns between real and synthetic data.")

    dist_scores: dict[str, float] = {}
    autocorr_scores: dict[str, float] = {}

    for col in common_cols:
        r_col = real_data[col]
        s_col = synthetic_data[col]

        # Distribution similarity
        dist_scores[col] = _mean_std_similarity(r_col, s_col)

        # Autocorrelation similarity
        r_ac = _sequence_autocorrelation(real_data, sequence_key, col)
        s_ac = _sequence_autocorrelation(synthetic_data, sequence_key, col)

        # Scale difference to [0, 1] where 1 is identical
        ac_diff = abs(r_ac - s_ac)
        autocorr_scores[col] = float(max(0.0, 1.0 - ac_diff))

    all_scores = list(dist_scores.values()) + list(autocorr_scores.values())
    overall = float(np.mean(all_scores)) if all_scores else 0.0

    result: dict[str, Any] = {
        "mean_std_similarity": dist_scores,
        "autocorr_similarity": autocorr_scores,
        "overall_score": overall,
    }

    if metadata is not None:
        try:
            from sdv.evaluation.single_table import evaluate_quality

            sdv_report = evaluate_quality(real_data, synthetic_data, metadata, verbose=False)
            result["sdv_quality_score"] = sdv_report.get_score()
        except Exception:
            pass

    return result
