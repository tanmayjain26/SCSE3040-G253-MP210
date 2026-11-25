import pandas as pd

from tea.checks import (duplicate_rows, leaking_column, spelling_variants,
                        stuck_sensor, target_typos, unit_mixup)


def make_df(n=40):
    rows = []
    for i in range(n):
        section = ["Lower Garden", "River Side", "Upper Garden"][i % 3]
        rain = 150.0 + i if section != "Upper Garden" else 15.0 + i / 10
        weight = 20.0 + i / 2
        rows.append({
            "record_id": f"R{i}", "date": f"2026-01-{i % 28 + 1:02d}",
            "worker_exp_yr": i % 30, "rain_mm_week": rain, "slope_deg": i % 35,
            "temp_c": 15.0 + i / 3, "bush_age_yr": 10 + i, "hours_worked": 5 + i / 10,
            "flush": "monsoon", "section": section,
            "kg_weighed_at_factory": weight, "kg_per_worker": weight + 0.1,
        })
    return pd.DataFrame(rows)


def test_leaking_column_is_the_one_missing_from_holdout():
    df = make_df()
    holdout = df.drop(columns=["kg_weighed_at_factory", "kg_per_worker"])
    result = leaking_column(df, holdout)
    assert result == {"present": True, "column": "kg_weighed_at_factory"}


def test_unit_mixup_finds_the_section_and_factor():
    result = unit_mixup(make_df())
    assert result["present"] is True
    assert result["column"] == "rain_mm_week"
    assert result["group_value"] == "Upper Garden"
    assert result["factor"] == 0.1


def test_duplicate_rows_counts_extra_copies():
    df = make_df()
    with_copies = pd.concat([df, df.iloc[:3]], ignore_index=True)
    assert duplicate_rows(with_copies)["count"] == 3
    assert duplicate_rows(df)["present"] is False


def test_spelling_variants_reports_true_levels():
    df = make_df()
    df.loc[0, "flush"] = "MON."
    df.loc[1, "flush"] = "monsoon "
    df.loc[2, "flush"] = "first"
    result = spelling_variants(df)
    assert result["column"] == "flush"
    assert result["true_levels"] == 2


def test_target_typos_counts_ten_times_too_big():
    df = make_df()
    df.loc[[2, 5], "kg_per_worker"] *= 10
    assert target_typos(df)["count"] == 2


def test_stuck_sensor_finds_a_long_constant_run():
    df = make_df(n=60)
    df["date"] = pd.date_range("2026-01-01", periods=60).strftime("%Y-%m-%d")
    df.loc[10:30, "temp_c"] = 25.0
    result = stuck_sensor(df)
    assert result["present"] is True
    assert result["column"] == "temp_c"
    assert result["start_date"] == "2026-01-11"
    assert result["end_date"] == "2026-01-31"
