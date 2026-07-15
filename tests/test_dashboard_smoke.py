"""Smoke test: the Streamlit app runs headless without raising."""
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest


def test_app_runs_without_exception():
    at = AppTest.from_file("dashboard/app.py", default_timeout=30).run()
    assert not at.exception
    assert len(at.metric) == 3  # min MAP, minutes<65, Ce


def test_higher_propofol_lowers_min_map():
    at = AppTest.from_file("dashboard/app.py", default_timeout=30).run()
    prop = [s for s in at.slider if s.label == "Infusion (mg/min)"][0]
    low_min = at.metric[0].value
    prop.set_value(80).run()
    high_min = at.metric[0].value
    # metric strings like "68 mmHg" -> compare numeric prefix
    assert int(high_min.split()[0]) <= int(low_min.split()[0])


def test_frail_preset_lowers_min_map():
    at = AppTest.from_file("dashboard/app.py", default_timeout=30).run()
    base_min = int(at.metric[0].value.split()[0])
    frail = [b for b in at.button if b.label == "Frail elderly"][0]
    frail.click().run()
    frail_min = int(at.metric[0].value.split()[0])
    assert frail_min <= base_min
