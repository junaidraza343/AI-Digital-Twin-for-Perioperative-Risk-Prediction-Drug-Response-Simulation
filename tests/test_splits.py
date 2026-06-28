import pandas as pd
import twin.config as c
from twin.data.splits import make_splits


def _meta(n=200):
    return pd.DataFrame({
        "caseid": range(n),
        "age": [20 + (i % 60) for i in range(n)],
        "asa": [1 + (i % 3) for i in range(n)],
        "preop_htn": [bool(i % 2) for i in range(n)],
    })


def test_every_case_assigned_one_split():
    assign = make_splits(_meta())
    assert set(assign.values()) <= {"train", "val", "test"}
    assert len(assign) == 200


def test_no_leakage_caseids_disjoint():
    assign = make_splits(_meta())
    buckets = {"train": set(), "val": set(), "test": set()}
    for cid, sp in assign.items():
        buckets[sp].add(cid)
    assert buckets["train"].isdisjoint(buckets["val"])
    assert buckets["train"].isdisjoint(buckets["test"])
    assert buckets["val"].isdisjoint(buckets["test"])


def test_ratios_approximately_hold():
    assign = make_splits(_meta(1000))
    frac_train = sum(v == "train" for v in assign.values()) / 1000
    assert abs(frac_train - c.SPLIT_RATIOS[0]) < 0.05


def test_deterministic_with_seed():
    assert make_splits(_meta()) == make_splits(_meta())
