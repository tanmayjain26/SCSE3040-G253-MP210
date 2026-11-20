"""Seven data-problem detectors for train.csv (MP210).

Each function takes the raw DataFrame and returns a dict whose keys match
the matching block in reports/findings.yaml. Owner is noted per function.
"""
import pandas as pd
import yaml


def load_params(path="params.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def load_train(params=None):
    params = params or load_params()
    return pd.read_csv(params["data"]["train"])


def leaking_column(df, holdout):          # owner: Samya
    # TODO: columns missing from holdout; which one tracks the target?
    return {"present": None, "column": None}


def target_typos(df):                     # owner: Samya
    # TODO: targets ~10x too big; return how many
    return {"present": None, "count": None}


def unit_mixup(df):                       # owner: Shyam
    """Find a numeric column that one section recorded in another unit.

    Idea: for each numeric column compute the mean per section. If one section's
    mean is about 10x / 12x / 60x / 100x / 1000x bigger or smaller than the mean
    of the other sections, that section used a different unit.
    factor is "wrong value = right value x factor" (so 0.1 means the section's
    numbers are 10 times too small).
    """
    candidates = [10, 12, 60, 100, 1000]
    candidates = candidates + [1 / c for c in candidates]
    num_cols = df.select_dtypes("number").columns
    best = None
    for col in num_cols:
        means = df.groupby("section")[col].mean()
        for section, m in means.items():
            others = means.drop(section).mean()
            if others == 0 or m == 0:
                continue
            ratio = m / others
            if 0.2 < ratio < 5:           # about the same size -> normal
                continue
            # snap the rough ratio to the nearest exact unit conversion
            factor = min(candidates, key=lambda c: abs(np.log(ratio) - np.log(c)))
            gap = abs(np.log(ratio) - np.log(factor))
            if best is None or gap < best[0]:
                best = (gap, col, section, factor)
    if best is None:
        return {"present": False, "column": None, "group_value": None, "factor": None}
    _, col, section, factor = best
    return {"present": True, "column": col, "group_value": section, "factor": factor}


def sentinel_value(df):                   # owner: Shyam
    """Find a special number (-1, -999, 999, 9999 ...) typed instead of a blank.

    Idea: for every numeric column count how many rows hold each of the usual
    sentinel numbers. Real data rarely contains -999 or 9999, so a clearly
    impossible number that repeats is a sentinel.
    """
    sentinels = [-9999, -999, -99, -1, 999, 9999]
    best = None
    for col in df.select_dtypes("number").columns:
        for value in sentinels:
            count = int((df[col] == value).sum())
            if count >= 3 and (best is None or count > best[2]):
                best = (col, value, count)
    if best is None:
        return {"present": False, "column": None, "value": None, "count": None}
    col, value, count = best
    return {"present": True, "column": col, "value": value, "count": count}


def duplicate_rows(df):                   # owner: Tanish
    # TODO: number of EXTRA copies
    return {"present": None, "count": None}


def stuck_sensor(df):                     # owner: Tanish
    # TODO: sort by date; long run of one identical value in a column
    return {"present": None, "column": None, "start_date": None, "end_date": None}


def spelling_variants(df):                # owner: Prashant
    # TODO: text columns; number of real categories after cleaning
    return {"present": None, "column": None, "true_levels": None}


def run_all():
    params = load_params()
    df = load_train(params)
    holdout = pd.read_csv(params["data"]["holdout"])
    results = {
        "leaking_column": leaking_column(df, holdout),
        "unit_mixup": unit_mixup(df),
        "sentinel_value": sentinel_value(df),
        "duplicate_rows": duplicate_rows(df),
        "stuck_sensor": stuck_sensor(df),
        "target_typos": target_typos(df),
        "spelling_variants": spelling_variants(df),
    }
    for name, res in results.items():
        print(name, res)
    return results


if __name__ == "__main__":
    run_all()