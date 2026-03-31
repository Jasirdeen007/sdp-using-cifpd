from __future__ import annotations

import pandas as pd

from src.sdp_pipeline import (
    TARGET_COLUMN,
    build_intent_text,
    get_candidate_feature_columns,
    load_dataset,
)


def test_load_dataset_creates_target_and_drops_null_rs(tmp_path):
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text(
        "bugID;rs;sd;bsr\n"
        "1;FIXED;Null pointer error;high\n"
        "2;;Missing status;low\n"
        "3;WONTFIX;UI alignment bug;medium\n"
    )

    df = load_dataset(csv_path)

    assert TARGET_COLUMN in df.columns
    assert len(df) == 2
    assert df[TARGET_COLUMN].tolist() == [1, 0]


def test_candidate_features_remove_leakage_column():
    df = pd.DataFrame(
        {
            "bugID": [1],
            "rs": ["FIXED"],
            "sd": ["Issue summary"],
            "bsr": ["high"],
            "defective": [1],
        }
    )

    features = get_candidate_feature_columns(df)

    assert "rs" not in features
    assert "bugID" not in features
    assert "sd" in features


def test_build_intent_text_joins_columns_without_blank_noise():
    df = pd.DataFrame(
        {
            "sd": ["Crash on startup"],
            "bsr": [""],
            "component": ["core"],
        }
    )

    intent = build_intent_text(df, ["sd", "bsr", "component"])

    assert intent.iloc[0] == "Crash on startup core"
