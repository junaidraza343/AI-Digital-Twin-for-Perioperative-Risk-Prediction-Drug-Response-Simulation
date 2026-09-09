# Perioperative Digital Twin — what-if MAP monitor with integrated predictor.
# FastAPI + custom web UI over the mechanistic PK-PD engine, the trained IOH
# predictor, data-driven calibration, and real VitalDB cases (bundled, offline).
# Cross-platform: builds and runs on macOS, Linux, and Windows (Docker Desktop,
# Linux containers). No shell scripts, JSON-array CMD, no host bind mounts.
#
# Ships the full stack: clinical monitor AND a live ML-Ops console (metrics,
# curves, retrain, VitalDB dataset build), so the deployed demo is the same
# application as the local one.
FROM python:3.10-slim

WORKDIR /app

# lightgbm needs libgomp at runtime; without it `import lightgbm` raises and
# GbdtModel silently degrades to the HistGradientBoosting fallback.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Pinned: an unpinned image silently changes model behaviour between builds,
# so a deployment could stop reproducing the numbers reported in the write-up.
# These match the versions the results were generated with.
# Large wheels (pyarrow, scikit-learn, lightgbm) time out on a loaded network;
# without explicit retries the whole image build fails on one slow download.
RUN pip install --no-cache-dir --retries 10 --timeout 120 \
    "numpy==1.26.4" \
    "pandas==2.3.1" \
    "pyarrow==19.0.1" \
    "scikit-learn==1.7.2" \
    "lightgbm==4.5.0" \
    "joblib==1.2.0" \
    "fastapi==0.115.6" \
    "uvicorn[standard]==0.51.0" \
    "vitaldb==1.7.2" \
    "certifi==2026.2.25"

# App code. scripts/ is required: the retrain and dataset-build endpoints import
# scripts.run_baseline / scripts.train_predictor and spawn scripts.run_all_subset.
COPY twin/ ./twin/
COPY api/ ./api/
COPY frontend/ ./frontend/
COPY scripts/ ./scripts/

# Runtime artifacts. windows.parquet + eligible_cases.csv drive the ML-Ops
# console (status counts, ROC/PR/calibration curves, live retrain); the CSVs
# back the selection-bias table and decision curve.
COPY results/windows.parquet results/eligible_cases.csv \
     results/predictor_map_only.joblib results/predictor_meta.json \
     results/demo_cases.json results/selection_bias_results.csv \
     results/deepnet_results.csv results/decision_curve.csv \
     results/cohort_funnel.csv ./results/

# Real VitalDB cases for offline grounding — only the 40 curated demo cases the
# UI can actually select, not the full 113 MB dev cache.
COPY deploy/demo_cache/ ./data_cache/

# Hugging Face Spaces (and most free hosts) run the container as a non-root UID.
# results/ and data_cache/ must stay writable: retrain rewrites the predictor and
# the build job streams logs + downloaded cases into them.
RUN useradd -m -u 1000 twin && chown -R twin:twin /app
USER twin

# Admin controls (retrain, dataset build) spawn compute and are DISABLED unless
# ADMIN_TOKEN is set in the environment; callers must then send it as
# X-Admin-Token. Deliberately NOT declared as ENV here: baking a credential name
# into image metadata is what SecretsUsedInArgOrEnv warns about, and an unset
# variable already gives the read-only default. Supply it at run time:
#   docker run -e ADMIN_TOKEN="$(openssl rand -hex 24)" ...

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
