from __future__ import annotations

import pandas as pd
import pytest

from utils.global_position_classifier import split_cross_session, split_dataframe


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trial": ["01", "01", "02", "02"],
            "user": ["01", "02", "01", "03"],
            "location": ["A-1", "A-2", "A-1", "A-3"],
            "group_id": ["a", "b", "c", "d"],
        }
    )


def test_cross_session_uses_all_trial_01_and_only_trial_02_for_test() -> None:
    train_df, test_df = split_cross_session(_frame())

    assert train_df["trial"].tolist() == ["01", "01"]
    assert test_df["trial"].tolist() == ["02", "02"]
    assert set(train_df["user"]) == {"01", "02"}
    assert set(test_df["user"]) == {"01", "03"}


def test_cross_session_is_available_through_general_splitter() -> None:
    train_df, test_df = split_dataframe(_frame(), split_mode="cross_session")
    assert len(train_df) == 2
    assert len(test_df) == 2


def test_cross_session_requires_trial_02() -> None:
    with pytest.raises(ValueError, match="no trial-02 test windows"):
        split_cross_session(_frame().loc[lambda frame: frame["trial"] == "01"])
