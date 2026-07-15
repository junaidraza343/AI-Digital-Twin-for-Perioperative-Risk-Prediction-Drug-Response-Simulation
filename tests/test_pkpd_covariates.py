import math
from twin.pkpd.params import POP_PROPOFOL
from twin.pkpd.covariates import scale_propofol


def test_reference_patient_unchanged():
    # 35 yo, 70 kg reference -> volumes unchanged, clearances ~unchanged
    p = scale_propofol(POP_PROPOFOL, age=35, weight=70, height=170, sex="M")
    assert math.isclose(p.V1, POP_PROPOFOL.V1, rel_tol=1e-9)
    assert math.isclose(p.CL, POP_PROPOFOL.CL, rel_tol=0.05)


def test_volumes_scale_linearly_with_weight():
    p = scale_propofol(POP_PROPOFOL, age=35, weight=140, height=170, sex="M")
    assert math.isclose(p.V1, POP_PROPOFOL.V1 * 2.0, rel_tol=1e-9)


def test_clearance_scales_allometrically():
    p = scale_propofol(POP_PROPOFOL, age=35, weight=140, height=170, sex="M")
    assert math.isclose(p.CL, POP_PROPOFOL.CL * (140 / 70) ** 0.75, rel_tol=1e-6)


def test_older_patient_lower_clearance():
    young = scale_propofol(POP_PROPOFOL, age=35, weight=70, height=170, sex="M")
    old = scale_propofol(POP_PROPOFOL, age=80, weight=70, height=170, sex="M")
    assert old.CL < young.CL
