# Perioperative Digital Twin — what-if MAP monitor with integrated predictor.
# FastAPI + custom web UI over the mechanistic PK-PD engine, the trained IOH
# predictor, data-driven calibration, and real VitalDB cases (bundled, offline).
# Cross-platform: builds and runs on macOS, Linux, and Windows (Docker Desktop,
# Linux containers). No shell scripts, JSON-array CMD, no host bind mounts.
FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    "numpy>=1.24" \
    "pandas>=2.0" \
    "pyarrow>=14.0" \
    "scikit-learn>=1.3" \
    "joblib>=1.2" \
    "fastapi>=0.110" \
    "uvicorn[standard]>=0.27"

# App code.
COPY twin/ ./twin/
COPY api/ ./api/
COPY frontend/ ./frontend/

# Runtime artifacts: trained predictor + real-case data for offline grounding.
COPY data_cache/ ./data_cache/
COPY results/predictor_map_only.joblib results/predictor_meta.json \
     results/demo_cases.json results/eligible_cases.csv ./results/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
