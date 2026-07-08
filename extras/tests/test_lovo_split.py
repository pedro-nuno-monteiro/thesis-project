from __future__ import annotations

import pandas as pd

from utils.global_position_classifier import split_lovo_folds


def test_split_lovo_folds_holds_out_one_user_at_a_time() -> None:
    df = pd.DataFrame(
        {
            "user": ["1", "1", "2", "2", "3", "3"],
            "location": ["A-1", "A-2", "B-1", "B-2", "C-1", "C-2"],
        }
    )

    folds = split_lovo_folds(df)

    assert len(folds) == 3
    held_out_users = []
    for train_df, test_df in folds:
        test_users = set(test_df["user"])
        train_users = set(train_df["user"])
        assert len(test_users) == 1
        assert test_users.isdisjoint(train_users)
        held_out_users.extend(test_users)

    assert sorted(held_out_users) == ["1", "2", "3"]
