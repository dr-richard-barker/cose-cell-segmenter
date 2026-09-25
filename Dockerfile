FROM python:3.12-slim

WORKDIR /app

# Install uv for fast dependency resolution and system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential libgl1 libglib2.0-0 \
    && curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="/usr/local/bin" sh \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
COPY src/ ./src/

# Install dependencies using uv
# We don't strictly need Qt (napari) but it's in pyproject.toml, so uv sync will install it.
# If we wanted a leaner image, we could strip out napari from a production pyproject.toml, 
# but this is fine for now.
RUN uv sync --no-dev --frozen

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "src"]
