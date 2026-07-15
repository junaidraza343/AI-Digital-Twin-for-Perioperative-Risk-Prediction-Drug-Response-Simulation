# What-if digital-twin dashboard — minimal CPU image.
FROM python:3.10-slim

WORKDIR /app

# Only the deps the mechanistic engine + dashboard need (not the SP1 data stack).
RUN pip install --no-cache-dir \
    "numpy>=1.24" \
    "matplotlib>=3.7" \
    "streamlit>=1.30"

# App code: the PK-PD engine package and the Streamlit UI.
COPY twin/ ./twin/
COPY dashboard/ ./dashboard/

EXPOSE 8501

# Lightweight healthcheck against Streamlit's own health endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health').status==200 else 1)"

ENTRYPOINT ["streamlit", "run", "dashboard/app.py", \
            "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
