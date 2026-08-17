"""Temporal (rolling-origin) train/validation/test splitting.

Random splitting of football matches leaks future information into the
past and is forbidden in this project. All evaluation uses rolling-origin
(expanding window) splits over a chronologically ordered match table.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class Fold:
    """One rolling-origin fold.

    train_idx / test_idx are positional indices into the chronological
    match table. Validation is carved from the tail of the training
    region for hyperparameter tuning.
    """

    fold_id: int
    train_idx: pd.Index
    val_idx: pd.Index
    test_idx: pd.Index


def rolling_origin_folds(
    df: pd.DataFrame,
    n_folds: int = 4,
    test_fraction: float = 0.2,
    val_fraction_of_train: float = 0.1,
    min_train_matches: int = 1000,
) -> list[Fold]:
    """Build expanding-window folds over the final `test_fraction` of matches.

    The test region [T0, N) is divided into `n_folds` contiguous chunks.
    Fold k trains on everything before chunk k, validates on a tail slice
    of that training region, and tests on chunk k. Every test match is
    evaluated exactly once, always with models trained only on earlier
    matches.
    """
    n = len(df)
    if n < min_train_matches + 10:
        raise ValueError(
            f"Too few matches ({n}) for rolling-origin evaluation "
            f"(min_train_matches={min_train_matches})."
        )

    t0 = int(n * (1.0 - test_fraction))
    if t0 < min_train_matches:
        t0 = min_train_matches

    test_region = n - t0
    if test_region < n_folds:
        raise ValueError("Test region too small for the requested number of folds.")

    fold_size = test_region // n_folds
    folds: list[Fold] = []

    for k in range(n_folds):
        test_start = t0 + k * fold_size
        test_end = t0 + (k + 1) * fold_size if k < n_folds - 1 else n

        train_end = test_start
        n_val = max(1, int(train_end * val_fraction_of_train))
        val_idx = pd.RangeIndex(train_end - n_val, train_end)
        train_idx = pd.RangeIndex(0, train_end - n_val)

        folds.append(
            Fold(
                fold_id=k,
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=pd.RangeIndex(test_start, test_end),
            )
        )

    return folds


def assert_no_temporal_leakage(folds: list[Fold]) -> None:
    """Sanity check: every fold's training data strictly precedes its test data."""
    for f in folds:
        assert f.train_idx.max() < f.test_idx.min(), (
            f"Fold {f.fold_id}: train indices overlap or exceed test region."
        )
        assert f.val_idx.max() < f.test_idx.min(), (
            f"Fold {f.fold_id}: validation indices overlap test region."
        )