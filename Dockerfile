# updated from python 3.11 to 3.13
FROM python:3.13-slim

# xgboost needs the OpenMP runtime, which slim images do not ship
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Dependencies first, so code changes do not re-install the world.
# README.md rides along because pyproject.toml declares it as the
# package readme, and the package cannot build without it.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src/ src/
RUN uv sync --frozen --no-dev

# The model artifact ships inside the image; the service never trains
COPY models/model.json models/model.json

EXPOSE 8000
CMD ["uv", "run", "--no-sync", "uvicorn", "trial_conversion_model.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
