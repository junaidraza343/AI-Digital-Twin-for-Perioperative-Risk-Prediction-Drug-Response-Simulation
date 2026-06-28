import pandas as pd
import twin.config as c
from scripts.run_cohort import build_cohort


def test_build_cohort_writes_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(c, "RESULTS_DIR", tmp_path)
    fake_cases = pd.DataFrame([
        dict(caseid=1, age=40, department="General surgery", ane_type="General",
             casestart=0, caseend=4000, asa=2, preop_sbp=120, preop_dbp=80,
             preop_htn=False),
        dict(caseid=2, age=10, department="General surgery", ane_type="General",
             casestart=0, caseend=4000, asa=2, preop_sbp=120, preop_dbp=80,
             preop_htn=False),
    ])
    eligible, funnel = build_cohort(cases_df=fake_cases)
    assert list(eligible.caseid) == [1]
    assert (tmp_path / "cohort_funnel.csv").exists()
    assert (tmp_path / "eligible_cases.csv").exists()
