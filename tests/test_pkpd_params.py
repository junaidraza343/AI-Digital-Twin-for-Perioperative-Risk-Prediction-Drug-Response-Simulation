from twin.pkpd.params import PropofolParams, NorepiParams, PDParams, POP_PROPOFOL, POP_PD, POP_NOREPI


def test_population_propofol_values():
    p = POP_PROPOFOL
    assert p.V1 == 6.3 and p.V2 == 25.0 and p.V3 == 270.0
    assert p.CL == 1.8 and p.Q2 == 1.7 and p.Q3 == 0.84
    assert p.ke0 == 0.146


def test_population_pd_values():
    assert POP_PD.emax == 0.30 and POP_PD.ec50 == 4.0 and POP_PD.gamma == 2.5


def test_population_norepi_values():
    assert POP_NOREPI.tpeak_s == 74.0 and POP_NOREPI.dmap_max == 0.24


def test_params_are_frozen_dataclasses():
    import dataclasses
    assert dataclasses.is_dataclass(PropofolParams)
    p = POP_PROPOFOL
    try:
        p.V1 = 1.0
        raised = False
    except dataclasses.FrozenInstanceError:
        raised = True
    assert raised
