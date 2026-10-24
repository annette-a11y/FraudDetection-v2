# Fraud Detection Dashboard (v2, risk weights 1-10)
# On start: scores transactions.csv with the rule engine, rebuilds dashboard.html,
# then serves the dashboard on $PORT (default 8080).
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PORT=8080
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY fraud_model.py dashboard_template.html transactions.csv ./

RUN useradd --create-home --uid 10001 appuser && chown -R appuser /app
USER appuser

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/dashboard.html', timeout=4)"

CMD ["sh", "-c", "python fraud_model.py --input transactions.csv && exec python -m http.server ${PORT} --bind 0.0.0.0"]
