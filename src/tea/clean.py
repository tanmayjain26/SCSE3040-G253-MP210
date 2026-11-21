"""Cleaning functions for train.csv (MP210).

What we found in checks.py and fix here:
  * unit_mixup        -> Upper Garden recorded rain in cm, not mm: multiply by 10
  * spelling_variants -> `flush` is written in many ways: map to first/second/monsoon
  * leaking_column    -> kg_weighed_at_factory is only known after the day: drop it
"""
import pandas as pd

TARGET = "kg_per_worker"
LEAK_COLUMN = "kg_weighed_at_factory"
FEATURES = ["worker_exp_yr", "rain_mm_week", "slope_deg", "temp_c",
            "bush_age_yr", "hours_worked", "flush", "section"]
FLUSH_LEVELS = ["first", "second", "monsoon"]
UNIT_FIX = {"column": "rain_mm_week", "section": "Upper Garden", "factor": 10}


def clean_flush(value):
    """' MON. ' -> 'monsoon', 'FIR.' -> 'first', 'Second ' -> 'second'.

    Lower-case, remove spaces and a trailing dot, then match the start of
    one of the real levels (so 'mon' and 'fir' become 'monsoon' and 'first').
    """
    text = str(value).strip().lower().rstrip(".")
    for level in FLUSH_LEVELS:
        if level.startswith(text) and text != "":
            return level
    return text


def clean_train(df):
    """Full cleaning for train.csv: fix spellings, fix the unit mix-up, drop duplicates."""
    out = df.copy()
    out["flush"] = out["flush"].map(clean_flush)
    fix = (out["section"] == UNIT_FIX["section"])
    out.loc[fix, UNIT_FIX["column"]] = out.loc[fix, UNIT_FIX["column"]] * UNIT_FIX["factor"]
    out = out.drop_duplicates(subset=[c for c in out.columns if c != "record_id"])
    return out.reset_index(drop=True)


def prepare_features(df):
    """For later.csv, holdout and API requests: they are already in the right units,
    we only normalise the flush spelling and keep the 8 feature columns."""
    out = df.copy()
    out["flush"] = out["flush"].map(clean_flush)
    return out[FEATURES]


def split_xy(df):
    """Return X (the 8 features, leak column left out) and y (the target)."""
    return df[FEATURES], df[TARGET]
