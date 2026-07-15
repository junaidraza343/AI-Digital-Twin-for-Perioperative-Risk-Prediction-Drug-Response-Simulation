# Perioperative Digital Twin — what-if MAP monitor.
# FastAPI + custom web UI over the mechanistic PK-PD engine.
# Cross-platform: builds and runs on macOS, Linux, and Windows (Docker Desktop,
# Linux containers). No shell scripts, JSON-array CMD, no host bind mounts.
FROM python:3.10-slim

WORKDIR /app

# Only the deps the engine + API need (not the SP1 data stack).
RUN pip install --no-cache-dir \
    "numpy>=1.24" \
    "fastapi>=0.110" \
    "uvicorn[standard]>=0.27"

# App code: PK-PD engine package, API, and the static frontend.
COPY twin/ ./twin/
COPY api/ ./api/
COPY frontend/ ./frontend/

EXPOSE 8000

# Health endpoint served by the app; portable Python check (no curl needed).
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
