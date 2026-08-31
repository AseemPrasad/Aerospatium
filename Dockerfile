FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System dependencies for geospatial packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    gdal-bin \
    libgdal-dev \
    libproj-dev \
    proj-data \
    proj-bin \
    libgeos-dev \
    libspatialindex-dev \
    && rm -rf /var/lib/apt/lists/*

# Install GDAL Python bindings
ENV CPLUS_INCLUDE_DIR=/usr/include/gdal
ENV C_INCLUDE_DIR=/usr/include/gdal
ENV GDAL_VERSION=3.6.2

# Copy dependency specification
COPY pyproject.toml .
COPY README.md .

# Install Python dependencies
RUN pip install --upgrade pip

# Install core scientific stack first (faster cache)
RUN pip install \
    numpy>=1.24,<2.2 \
    pandas>=2.0 \
    scipy>=1.10 \
    pyarrow>=12.0 \
    duckdb>=0.9 \
    geopandas>=0.13 \
    shapely>=2.0 \
    h3>=4.0 \
    rasterio \
    xarray>=2023.08 \
    netcdf4>=1.6

# Install ML stack
RUN pip install \
    torch>=2.0 \
    xgboost>=2.0

# Install API server and utilities
RUN pip install \
    fastapi>=0.100 \
    "uvicorn[standard]>=0.23" \
    pydantic[all]>=2.0 \
    pyyaml>=6.0

# Install dev dependencies
RUN pip install \
    pytest>=7.4 \
    pytest-cov>=4.1 \
    pytest-xdist>=3.3 \
    ruff>=0.1.0 \
    mypy>=1.5

# Copy application code
COPY src/ src/

# Create data directories
RUN mkdir -p data/raw data/processed models reports

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "aerospatium.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
