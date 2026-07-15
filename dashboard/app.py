"""What-if digital-twin dashboard: dose propofol/norepinephrine, watch projected MAP."""
import matplotlib.pyplot as plt
import streamlit as st

from twin.pkpd.engine import project_map, Patient

st.set_page_config(page_title="Perioperative Digital Twin — What-If", layout="wide")
st.title("Perioperative Digital Twin — What-If MAP Simulator")
st.caption("Mechanistic PK-PD engine (Eleveld propofol + Joachim norepinephrine). "
           "Decision support, not closed-loop control.")

with st.sidebar:
    st.header("Patient")
    age = st.slider("Age (yr)", 18, 90, 60)
    weight = st.slider("Weight (kg)", 40, 140, 70)
    height = st.slider("Height (cm)", 140, 200, 170)
    sex = st.radio("Sex", ["M", "F"], horizontal=True)
    map0 = st.slider("Baseline MAP (mmHg)", 60, 120, 90)
    duration = st.slider("Horizon (min)", 5, 30, 15) * 60.0

    st.header("Propofol")
    prop_rate = st.slider("Infusion (mg/min)", 0, 80, 30)

    st.header("Norepinephrine (rescue)")
    ne_rate = st.slider("Infusion (µg/min)", 0, 30, 0)
    ne_start = st.slider("Start (min)", 0, int(duration // 60), 0) * 60.0

    st.header("Personalization (calibration δ)")
    personalize = st.checkbox("Enable personalized twin", value=False)
    ec50_d = st.slider("δ EC50 (sensitivity)", -0.7, 0.7, 0.0, 0.05, disabled=not personalize)
    ke0_d = st.slider("δ ke0 (onset)", -0.7, 0.7, 0.0, 0.05, disabled=not personalize)


@st.cache_data
def run(age, weight, height, sex, map0, duration, prop_rate, ne_rate, ne_start, deltas):
    patient = Patient(age=age, weight=weight, height=height, sex=sex, map0=map0)
    prop = [(0.0, duration, float(prop_rate))] if prop_rate > 0 else []
    ne = [(ne_start, duration, float(ne_rate))] if ne_rate > 0 else []
    return project_map(patient, prop, ne, duration=duration, deltas=deltas or None)


pop = run(age, weight, height, sex, map0, duration, prop_rate, ne_rate, ne_start, None)
deltas = {"EC50": ec50_d, "ke0": ke0_d} if personalize else None
pers = run(age, weight, height, sex, map0, duration, prop_rate, ne_rate, ne_start, deltas)

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.axhspan(20, 65, color="red", alpha=0.08)
ax.axhline(65, color="red", ls="--", lw=1, label="IOH threshold (65 mmHg)")
ax.plot(pop.t / 60.0, pop.map, lw=2, label="Population twin")
if personalize:
    ax.plot(pers.t / 60.0, pers.map, lw=2, ls="-", color="darkorange", label="Personalized twin")
ax.set_xlabel("Time (min)")
ax.set_ylabel("Projected MAP (mmHg)")
ax.set_ylim(40, max(120, map0 + 10))
ax.legend(loc="upper right")
st.pyplot(fig)

shown = pers if personalize else pop
c1, c2, c3 = st.columns(3)
c1.metric("Min projected MAP", f"{shown.map.min():.0f} mmHg")
c2.metric("Minutes < 65", f"{shown.minutes_below_65:.1f} min")
c3.metric("Propofol Ce (end)", f"{shown.ce[-1]:.2f} µg/mL")
