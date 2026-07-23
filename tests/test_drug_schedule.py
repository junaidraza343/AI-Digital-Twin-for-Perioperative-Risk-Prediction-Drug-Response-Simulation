import numpy as np
import pandas as pd
import twin.config as c
from twin.data.drug_schedule import propofol_mg_min, vasopressor_ug_min


def _frame(n=10, **cols):
    base = {t: np.full(n, np.nan) for t in c.NUMERIC_TRACKS}
    base.update(cols)
    return pd.DataFrame(base)


def test_propofol_mg_min_converts_ml_per_hour():
    # 60 mL/h of 20 mg/mL propofol = 60*20/60 = 20 mg/min
    f = _frame(n=5, **{c.PROPOFOL_TRACK: np.full(5, 60.0)})
    out = propofol_mg_min(f)
    assert out.shape == (5,)
    assert np.allclose(out, 20.0)


def test_propofol_nan_becomes_zero():
    f = _frame(n=4, **{c.PROPOFOL_TRACK: [np.nan, 60.0, np.nan, 0.0]})
    out = propofol_mg_min(f)
    assert np.allclose(out, [0.0, 20.0, 0.0, 0.0])


def test_vasopressor_sums_phen_and_nepi_in_ug_min():
    # 60 mL/h PHEN @100 ug/mL = 100 ug/min ; 60 mL/h NEPI @20 = 20 ug/min ; sum=120
    f = _frame(n=3, **{c.PHENYLEPHRINE_TRACK: np.full(3, 60.0),
                       c.NOREPI_TRACK: np.full(3, 60.0)})
    out = vasopressor_ug_min(f)
    assert np.allclose(out, 120.0)
