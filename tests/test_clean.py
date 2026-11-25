import pandas as pd

from tea.clean import FEATURES, clean_flush, clean_train, prepare_features, split_xy


def small_train():
    return pd.DataFrame({
        "record_id": ["R1", "R2", "R3", "R4"],
        "date": ["2026-01-01"] * 4,
        "worker_exp_yr": [10, 20, 30, 5],
        "rain_mm_week": [150.0, 15.0, 200.0, 18.0],
        "slope_deg": [5, 10, 15, 20],
        "temp_c": [20.0, 21.0, 22.0, 23.0],
        "bush_age_yr": [30, 40, 50, 60],
        "hours_worked": [6.0, 7.0, 8.0, 9.0],
        "flush": ["Monsoon ", "FIR.", "sec.", "second"],
        "section": ["Lower Garden", "Upper Garden", "River Side", "Upper Garden"],
        "kg_weighed_at_factory": [30.0, 31.0, 32.0, 33.0],
        "kg_per_worker": [29.9, 31.0, 32.1, 33.0],
    })


def test_clean_flush_maps_all_spellings():
    assert clean_flush(" MONSOON") == "monsoon"
    assert clean_flush("mon.") == "monsoon"
    assert clean_flush("FIR.") == "first"
    assert clean_flush("Second ") == "second"


def test_clean_train_fixes_upper_garden_rain_only():
    out = clean_train(small_train())
    upper = out[out["section"] == "Upper Garden"]["rain_mm_week"].tolist()
    lower = out[out["section"] == "Lower Garden"]["rain_mm_week"].tolist()
    assert upper == [150.0, 180.0]      # 15 and 18 became 150 and 180
    assert lower == [150.0]             # other sections unchanged
    assert set(out["flush"]) <= {"first", "second", "monsoon"}


def test_split_xy_drops_the_leaking_column():
    X, y = split_xy(clean_train(small_train()))
    assert list(X.columns) == FEATURES
    assert "kg_weighed_at_factory" not in X.columns
    assert len(X) == len(y)


def test_prepare_features_keeps_units_and_normalises_flush():
    df = small_train()
    out = prepare_features(df)
    assert list(out.columns) == FEATURES
    assert out["rain_mm_week"].tolist() == df["rain_mm_week"].tolist()  # no unit fix here
    assert out["flush"].tolist() == ["monsoon", "first", "second", "second"]
