"""What-if digital-twin dashboard: dose propofol/norepinephrine, watch projected MAP."""
import matplotlib.pyplot as plt
import streamlit as st

from twin.pkpd.engine import project_map, project_band, ioh_risk, Patient

st.set_page_config(page_title="Perioperative Digital Twin — What-If", layout="wide")
st.title("Perioperative Digital Twin — What-If MAP Simulator")
st.caption("Mechanistic PK-PD engine (Eleveld propofol + Joachim norepinephrine). "
           "Decision support, not closed-loop control.")

# --- preset patient scenarios -------------------------------------------------
PRESETS = {
    "Healthy adult": dict(age=35, weight=75, height=175, sex="M", map0=95, prop_rate=25),
    "Frail elderly": dict(age=82, weight=54, height=162, sex="F", map0=82, prop_rate=45),
}
_DEFAULTS = dict(age=60, weight=70, height=170, sex="M", map0=90, prop_rate=30,
                 ne_rate=0, ne_start=0, personalize=False, ec50_d=0.0, ke0_d=0.0,
                 show_band=True)
for _k, _v in _DEFAULTS.items():
    st.session_state.setdefault(_k, _v)


def apply_preset(name):
    for k, v in PRESETS[name].items():
        st.session_state[k] = v


with st.sidebar:
    st.header("Preset scenarios")
    pc1, pc2 = st.columns(2)
    pc1.button("Healthy adult", on_click=apply_preset, args=("Healthy adult",),
               use_container_width=True)
    pc2.button("Frail elderly", on_click=apply_preset, args=("Frail elderly",),
               use_container_width=True)

    st.header("Patient")
    age = st.slider("Age (yr)", 18, 90, key="age")
    weight = st.slider("Weight (kg)", 40, 140, key="weight")
    height = st.slider("Height (cm)", 140, 200, key="height")
    sex = st.radio("Sex", ["M", "F"], horizontal=True, key="sex")
    map0 = st.slider("Baseline MAP (mmHg)", 60, 120, key="map0")
    duration = st.slider("Horizon (min)", 5, 30, 15) * 60.0

    st.header("Propofol")
    prop_rate = st.slider("Infusion (mg/min)", 0, 80, key="prop_rate")

    st.header("Norepinephrine (rescue)")
    ne_rate = st.slider("Infusion (µg/min)", 0, 30, key="ne_rate")
    ne_start = st.slider("Start (min)", 0, int(duration // 60), key="ne_start") * 60.0

    st.header("Personalization (calibration δ)")
    personalize = st.checkbox("Enable personalized twin", key="personalize")
    ec50_d = st.slider("δ EC50 (sensitivity)", -0.7, 0.7, step=0.05, key="ec50_d",
                       disabled=not personalize)
    ke0_d = st.slider("δ ke0 (onset)", -0.7, 0.7, step=0.05, key="ke0_d",
                      disabled=not personalize)

    st.header("Display")
    show_band = st.checkbox("Show confidence band", key="show_band")


@st.cache_data
def run(age, weight, height, sex, map0, duration, prop_rate, ne_rate, ne_start, deltas):
    patient = Patient(age=age, weight=weight, height=height, sex=sex, map0=map0)
    prop = [(0.0, duration, float(prop_rate))] if prop_rate > 0 else []
    ne = [(ne_start, duration, float(ne_rate))] if ne_rate > 0 else []
    return project_map(patient, prop, ne, duration=duration, deltas=deltas or None)


@st.cache_data
def run_band(age, weight, height, sex, map0, duration, prop_rate, ne_rate, ne_start, deltas):
    patient = Patient(age=age, weight=weight, height=height, sex=sex, map0=map0)
    prop = [(0.0, duration, float(prop_rate))] if prop_rate > 0 else []
    ne = [(ne_start, duration, float(ne_rate))] if ne_rate > 0 else []
    return project_band(patient, prop, ne, duration=duration, deltas=deltas or None)


deltas = {"EC50": ec50_d, "ke0": ke0_d} if personalize else None
pop = run(age, weight, height, sex, map0, duration, prop_rate, ne_rate, ne_start, None)
pers = run(age, weight, height, sex, map0, duration, prop_rate, ne_rate, ne_start, deltas)
shown = pers if personalize else pop

# --- IOH risk badge -----------------------------------------------------------
risk = ioh_risk(shown)
_risk_color = {"Low": "#1a9850", "Medium": "#f5a700", "High": "#d73027"}[risk]
st.markdown(
    f"<div style='padding:0.6rem 1rem;border-radius:0.5rem;background:{_risk_color};"
    f"color:white;font-weight:700;display:inline-block'>Predicted IOH risk: {risk}</div>",
    unsafe_allow_html=True,
)

# --- MAP projection plot ------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 4.2))
ax.axhspan(20, 65, color="red", alpha=0.08)
ax.axhline(65, color="red", ls="--", lw=1, label="IOH threshold (65 mmHg)")
if show_band:
    lo, hi, _ = run_band(age, weight, height, sex, map0, duration,
                         prop_rate, ne_rate, ne_start, deltas)
    ax.fill_between(shown.t / 60.0, lo, hi, color="steelblue", alpha=0.18,
                    label="Confidence band")
ax.plot(pop.t / 60.0, pop.map, lw=2, label="Population twin")
if personalize:
    ax.plot(pers.t / 60.0, pers.map, lw=2, color="darkorange", label="Personalized twin")
ax.set_xlabel("Time (min)")
ax.set_ylabel("Projected MAP (mmHg)")
ax.set_ylim(40, max(120, map0 + 10))
ax.legend(loc="upper right")
st.pyplot(fig)

c1, c2, c3 = st.columns(3)
c1.metric("Min projected MAP", f"{shown.map.min():.0f} mmHg")
c2.metric("Minutes < 65", f"{shown.minutes_below_65:.1f} min")
c3.metric("Propofol Ce (end)", f"{shown.ce[-1]:.2f} µg/mL")

# --- effect-site concentration ------------------------------------------------
with st.expander("Propofol effect-site concentration (Ce)", expanded=True):
    fig2, ax2 = plt.subplots(figsize=(9, 2.6))
    ax2.plot(shown.t / 60.0, shown.ce, color="purple", lw=1.8)
    ax2.set_xlabel("Time (min)")
    ax2.set_ylabel("Ce (µg/mL)")
    ax2.set_title("Effect-site concentration driving the MAP reduction")
    st.pyplot(fig2)
