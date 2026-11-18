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
    # TODO: per-section stats per numeric column; exact factor
    return {"present": None, "column": None, "group_value": None, "factor": None}


def sentinel_value(df):                   # owner: Shyam
    # TODO: impossible repeated value (-1, -999, 999, 9999 ...)
    return {"present": None, "column": None, "value": None, "count": None}


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