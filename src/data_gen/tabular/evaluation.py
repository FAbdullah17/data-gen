"""Evaluation metrics for tabular synthetic data.

Provides quality comparison between real and synthetic DataFrames
using statistical tests and distribution similarity measures.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mutual_info_score


def _ks_complement(
    real_col: pd.Series, synthetic_col: pd.Series  # type: ignore[type-arg]
) -> float:
    """Kolmogorov-Smirnov complement for a single numeric column.

    Returns a value between 0 and 1, where 1 means the distributions
    are identical and 0 means maximally different.
    """
    if real_col.nunique() <= 1 or synthetic_col.nunique() <= 1:
        return 1.0 if real_col.equals(synthetic_col) else 0.0

    statistic, _ = stats.ks_2samp(
        real_col.dropna().values,
        synthetic_col.dropna().values,
    )
    return 1.0 - statistic


def _cs_test(
    real_col: pd.Series, synthetic_col: pd.Series  # type: ignore[type-arg]
) -> float:
    """Chi-Squared test complement for a single categorical column.

    Returns a value between 0 and 1, where 1 means the distributions
    are identical and 0 means maximally different.
    """
    real_counts = real_col.value_counts(normalize=True)
    synthetic_counts = synthetic_col.value_counts(normalize=True)

    all_categories = sorted(set(real_counts.index) | set(synthetic_counts.index))
    real_freq = np.array([real_counts.get(c, 0.0) for c in all_categories])
    synth_freq = np.array([synthetic_counts.get(c, 0.0) for c in all_categories])

    if np.allclose(real_freq, synth_freq):
        return 1.0

    real_freq_nonzero = np.where(real_freq == 0, 1e-10, real_freq)
    chi2 = np.sum((synth_freq - real_freq) ** 2 / real_freq_nonzero)

    return max(0.0, 1.0 - chi2)


def _column_shape_score(
    real_col: pd.Series, synthetic_col: pd.Series  # type: ignore[type-arg]
) -> float:
    """Score how well the synthetic column matches the shape of the real one.

    For numeric columns: compares mean, std, min, max.
    For categorical columns: compares category frequency distribution.
    """
    if pd.api.types.is_numeric_dtype(real_col):
        real_stats = [real_col.mean(), real_col.std(), real_col.min(), real_col.max()]
        synth_stats = [
            synthetic_col.mean(),
            synthetic_col.std(),
            synthetic_col.min(),
            synthetic_col.max(),
        ]
        diffs = []
        for r, s in zip(real_stats, synth_stats):
            if r == 0 and s == 0:
                diffs.append(0.0)
            elif r == 0:
                diffs.append(1.0)
            else:
                diffs.append(min(abs(r - s) / (abs(r) + 1e-10), 1.0))
        return 1.0 - np.mean(diffs)

    return _cs_test(real_col, synthetic_col)


def _correlation_similarity(
    real_df: pd.DataFrame, synthetic_df: pd.DataFrame
) -> float:
    """Compare correlation matrices of real vs synthetic numeric columns.

    Returns a value between 0 and 1, where 1 means identical correlations.
    """
    numeric_cols = real_df.select_dtypes(include=np.number).columns.tolist()
    common_cols = [c for c in numeric_cols if c in synthetic_df.columns]

    if len(common_cols) < 2:
        return 1.0

    real_corr = real_df[common_cols].corr().values
    synth_corr = synthetic_df[common_cols].corr().values

    diff = np.abs(real_corr - synth_corr)
    return float(1.0 - np.nanmean(diff))


def evaluate_tabular(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    metadata: Any = None,
) -> dict[str, Any]:
    """Evaluate quality of synthetic tabular data against original.

    Parameters
    ----------
    real_data : pd.DataFrame
        Original sample data.
    synthetic_data : pd.DataFrame
        Generated synthetic data.
    metadata : Any, optional
        SDV metadata object. If provided and SDV is installed, runs
        SDV's built-in quality evaluation alongside custom metrics.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys:
        - ``ks_complement``: dict of per-column KS complement scores (numeric cols)
        - ``cs_test``: dict of per-column chi-squared scores (categorical cols)
        - ``column_shapes``: dict of per-column shape similarity scores
        - ``correlation_similarity``: overall correlation matrix similarity
        - ``overall_score``: weighted average of all metrics
        - ``sdv_quality`` (optional): SDV's built-in quality report if available

    Examples
    --------
    >>> metrics = evaluate_tabular(real_df, synthetic_df)
    >>> print(f"Overall quality: {metrics['overall_score']:.2%}")
    """
    if not isinstance(real_data, pd.DataFrame):
        raise TypeError(f"real_data must be a DataFrame, got {type(real_data).__name__}")
    if not isinstance(synthetic_data, pd.DataFrame):
        raise TypeError(
            f"synthetic_data must be a DataFrame, got {type(synthetic_data).__name__}"
        )

    common_cols = [c for c in real_data.columns if c in synthetic_data.columns]
    if not common_cols:
        raise ValueError("No common columns between real and synthetic data.")

    ks_scores: dict[str, float] = {}
    cs_scores: dict[str, float] = {}
    shape_scores: dict[str, float] = {}

    for col in common_cols:
        real_col = real_data[col]
        synth_col = synthetic_data[col]

        shape_scores[col] = _column_shape_score(real_col, synth_col)

        if pd.api.types.is_numeric_dtype(real_col):
            ks_scores[col] = _ks_complement(real_col, synth_col)
        else:
            cs_scores[col] = _cs_test(real_col, synth_col)

    corr_sim = _correlation_similarity(real_data, synthetic_data)

    all_scores = list(ks_scores.values()) + list(cs_scores.values()) + [corr_sim]
    overall = float(np.mean(all_scores)) if all_scores else 0.0

    result: dict[str, Any] = {
        "ks_complement": ks_scores,
        "cs_test": cs_scores,
        "column_shapes": shape_scores,
        "correlation_similarity": corr_sim,
        "overall_score": overall,
    }

    if metadata is not None:
        try:
            from sdv.evaluation.single_table import evaluate_quality

            sdv_report = evaluate_quality(
                real_data, synthetic_data, metadata, verbose=False
            )
            result["sdv_quality_score"] = sdv_report.get_score()
        except Exception:
            pass

    return result
