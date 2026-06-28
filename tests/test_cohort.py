import pandas as pd
from twin.data.cohort import filter_cohort


def _base_row(**over):
    row = dict(caseid=1, age=40, department="General surgery",
               ane_type="General", casestart=0, caseend=4000,
               asa=2, preop_sbp=120, preop_dbp=80, preop_htn=False)
    row.update(over)
    return row


def test_keeps_eligible_case():
    df = pd.DataFrame([_base_row()])
    eligible, funnel = filter_cohort(df)
    assert len(eligible) == 1
    assert funnel[0] == ("start", 1)


def test_excludes_minor():
    df = pd.DataFrame([_base_row(age=15)])
    eligible, _ = filter_cohort(df)
    assert len(eligible) == 0


def test_excludes_cardiac_department():
    df = pd.DataFrame([_base_row(department="Cardiac surgery")])
    eligible, _ = filter_cohort(df)
    assert len(eligible) == 0


def test_excludes_non_general_anesthesia():
    df = pd.DataFrame([_base_row(ane_type="Spinal")])
    eligible, _ = filter_cohort(df)
    assert len(eligible) == 0


def test_excludes_short_case():
    df = pd.DataFrame([_base_row(casestart=0, caseend=1000)])
    eligible, _ = filter_cohort(df)
    assert len(eligible) == 0


def test_excludes_asa_5():
    df = pd.DataFrame([_base_row(asa=5)])
    eligible, _ = filter_cohort(df)
    assert len(eligible) == 0


def test_excludes_preop_hypotension():
    # preop MAP = 50 + (60-50)/3 ~= 53 < 65
    df = pd.DataFrame([_base_row(preop_sbp=60, preop_dbp=50)])
    eligible, _ = filter_cohort(df)
    assert len(eligible) == 0


def test_funnel_records_each_step():
    df = pd.DataFrame([_base_row(), _base_row(caseid=2, age=10)])
    _, funnel = filter_cohort(df)
    steps = [name for name, _ in funnel]
    assert steps[0] == "start"
    assert "age>=18" in steps
