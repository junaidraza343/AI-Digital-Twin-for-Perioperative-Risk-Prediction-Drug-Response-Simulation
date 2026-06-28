# Central configuration. Constants only; no logic.
from pathlib import Path

SEED = 42
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_CACHE = REPO_ROOT / "data_cache"
RESULTS_DIR = REPO_ROOT / "results"

# Label definition
MAP_THRESHOLD = 65.0          # mmHg, hypotension threshold
EVENT_MIN_SECONDS = 60        # cumulative seconds below threshold to count as an event
HORIZON_SECONDS = 5 * 60      # prediction horizon
OBS_WINDOW_SECONDS = 10 * 60  # observation window length before t_end
STRIDE_SECONDS = 30           # window stride
HORIZON_MIN_VALID = 0.5       # min fraction of non-NaN samples required in horizon

# Gray-zone band (upper bound for "some dip" categorization)
GRAY_HIGH = 75.0

# 1 Hz numeric tracks to load from VitalDB
MAP_TRACK = "Solar8000/ART_MBP"
MAP_TRACK_FALLBACK = "Solar8000/NIBP_MBP"
NUMERIC_TRACKS = [
    MAP_TRACK, MAP_TRACK_FALLBACK,
    "Solar8000/ART_SBP", "Solar8000/ART_DBP",
    "Solar8000/HR", "Solar8000/PLETH_SPO2",
    "Solar8000/ETCO2", "Solar8000/RR", "Solar8000/BT",
]

# Cohort filtering
EXCLUDED_DEPARTMENTS = {"Cardiac surgery", "Thoracic surgery", "Gynecology"}
MIN_AGE = 18
MIN_DURATION_SECONDS = 1800
EXCLUDED_ASA = {5, 6}

# Splits
SPLIT_RATIOS = (0.70, 0.15, 0.15)

# Evaluation
ALARM_RATE = 0.10             # operating point for PPV reporting
ECE_BINS = 15
