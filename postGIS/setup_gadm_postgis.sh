#!/usr/bin/env bash
set -euo pipefail

# =========================
# Config
# =========================
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
SRC_PATH="${1:-"$SCRIPT_DIR/gadm_410.gdb"}"

PG_CONTAINER="gadm-postgis"
PG_DB="gis"
PG_USER="postgres"
PG_PASSWORD="postgres"
PG_PORT="5432"
PG_HOST="127.0.0.1"

PG_IMAGE="${PG_IMAGE:-postgis/postgis:16-3.4}"

# default to amd64 unless you override DOCKER_PLATFORM
HOST_ARCH="$(uname -m)"
DOCKER_PLATFORM="${DOCKER_PLATFORM:-}"
if [[ -z "$DOCKER_PLATFORM" && ( "$HOST_ARCH" == "arm64" || "$HOST_ARCH" == "aarch64" ) ]]; then
  DOCKER_PLATFORM="linux/amd64"
fi

PLATFORM_ARGS=()
if [[ -n "$DOCKER_PLATFORM" ]]; then
  PLATFORM_ARGS=(--platform "$DOCKER_PLATFORM")
  echo "Docker platform forced: $DOCKER_PLATFORM"
fi

# =========================
# Helpers
# =========================
retry() {
  local attempts="$1"; shift
  local sleep_s="$1"; shift
  local n=1
  until "$@"; do
    if (( n >= attempts )); then
      return 1
    fi
    sleep "$sleep_s"
    n=$((n+1))
  done
}

# =========================
# Checks
# =========================
if [[ ! -e "$SRC_PATH" ]]; then
  echo "ERROR: Source not found: $SRC_PATH"
  echo "Usage: $0 /absolute/path/to/gadm_410.gdb_or_gpkg"
  exit 1
fi

if [[ "$SRC_PATH" == *.gdb && ! -d "$SRC_PATH" ]]; then
  echo "ERROR: '$SRC_PATH' ends with .gdb but is not a directory."
  exit 1
fi

echo "Using source: $SRC_PATH"

command -v docker >/dev/null 2>&1 || { echo "ERROR: docker is required"; exit 1; }
command -v ogrinfo >/dev/null 2>&1 || { echo "ERROR: ogrinfo is required (brew install gdal)"; exit 1; }
command -v ogr2ogr >/dev/null 2>&1 || { echo "ERROR: ogr2ogr is required (brew install gdal)"; exit 1; }

# =========================
# Start PostGIS container
# =========================
if docker ps -a --format '{{.Names}}' | grep -qx "$PG_CONTAINER"; then
  echo "Container $PG_CONTAINER already exists. Starting it..."
  docker start "$PG_CONTAINER" >/dev/null || true
else
  echo "Creating PostGIS container $PG_CONTAINER"
  docker run -d \
    "${PLATFORM_ARGS[@]}" \
    --name "$PG_CONTAINER" \
    -e POSTGRES_DB="$PG_DB" \
    -e POSTGRES_USER="$PG_USER" \
    -e POSTGRES_PASSWORD="$PG_PASSWORD" \
    -p "$PG_PORT:5432" \
    --shm-size=1g \
    "$PG_IMAGE" >/dev/null
fi

sleep 1
if ! docker ps --format '{{.Names}}' | grep -qx "$PG_CONTAINER"; then
  echo "ERROR: container '$PG_CONTAINER' is not running. Recent logs:"
  docker logs "$PG_CONTAINER" --tail 200 || true
  exit 1
fi

# =========================
# Wait for Postgres
# =========================
echo "Waiting for PostgreSQL to be ready (inside container)..."
until docker exec "$PG_CONTAINER" pg_isready -U "$PG_USER" -d postgres >/dev/null 2>&1; do
  sleep 2
done
echo "PostgreSQL is ready (container check)."

# Ensure DB exists
echo "Ensuring database '$PG_DB' exists..."
if ! docker exec -e PGPASSWORD="$PG_PASSWORD" "$PG_CONTAINER" \
    psql -U "$PG_USER" -d postgres -tAc "SELECT 1 FROM pg_database WHERE datname='${PG_DB}'" | grep -q 1; then
  docker exec -e PGPASSWORD="$PG_PASSWORD" "$PG_CONTAINER" createdb -U "$PG_USER" "$PG_DB"
fi

# Ensure PostGIS
echo "Ensuring PostGIS extension exists..."
docker exec -e PGPASSWORD="$PG_PASSWORD" "$PG_CONTAINER" \
  psql -U "$PG_USER" -d "$PG_DB" -v ON_ERROR_STOP=1 -c "CREATE EXTENSION IF NOT EXISTS postgis;" >/dev/null

# =========================
# List layers
# =========================
echo "Reading layers from source (ogrinfo)..."
LAYER_NAMES="$(
  ogrinfo -ro "$SRC_PATH" 2>/dev/null \
    | awk -F': ' '
        /^[0-9]+: / {print $2}
        /^Layer: /   {print $2}
      ' \
    | sed -E 's/ \(.+$//' \
    | sed -E 's/[[:space:]]+$//' \
    | sort -u
)"

if [[ -z "$LAYER_NAMES" ]]; then
  echo "ERROR: No layers found."
  echo "Try: ogrinfo -ro \"$SRC_PATH\""
  exit 1
fi

echo "Found layers:"
echo "$LAYER_NAMES"

# =========================
# Host TCP sanity check
# =========================
PG_DSN="PG:host=${PG_HOST} port=${PG_PORT} dbname=${PG_DB} user=${PG_USER} password=${PG_PASSWORD}"

echo "Sanity check: waiting for host TCP connectivity via ogrinfo..."
if ! retry 30 2 ogrinfo -ro "$PG_DSN" -sql "select 1" >/dev/null 2>&1; then
  echo "ERROR: ogrinfo cannot connect to Postgres via ${PG_HOST}:${PG_PORT} after retries."
  echo "Container status:"
  docker ps -a --filter "name=^/${PG_CONTAINER}$" || true
  echo "Recent container logs:"
  docker logs "$PG_CONTAINER" --tail 200 || true
  exit 1
fi
echo "Host TCP connectivity looks good."

# =========================
# Import
# =========================
echo "Importing layers into PostGIS (this may take a while)..."

while IFS= read -r layer_name; do
  [[ -z "$layer_name" ]] && continue
  table_name="$(echo "$layer_name" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9_]/_/g')"

  echo
  echo "  -> Importing layer '$layer_name' as table 'public.${table_name}'"

  ogr2ogr \
    --config PG_USE_COPY YES \
    -f PostgreSQL \
    "$PG_DSN" \
    "$SRC_PATH" \
    "$layer_name" \
    -nln "$table_name" \
    -overwrite \
    -lco SCHEMA=public \
    -lco GEOMETRY_NAME=geom \
    -lco FID=gid \
    -nlt PROMOTE_TO_MULTI \
    -progress

  echo "     creating spatial index + ANALYZE..."
  docker exec -e PGPASSWORD="$PG_PASSWORD" "$PG_CONTAINER" \
    psql -U "$PG_USER" -d "$PG_DB" -v ON_ERROR_STOP=1 <<SQL >/dev/null
DO \$\$
DECLARE
  idx_name text := '${table_name}_geom_gix';
BEGIN
  IF EXISTS (
    SELECT 1
    FROM information_schema.columns
    WHERE table_schema = 'public'
      AND table_name = '${table_name}'
      AND column_name = 'geom'
  ) THEN
    EXECUTE format('CREATE INDEX IF NOT EXISTS %I ON public.%I USING GIST (geom)', idx_name, '${table_name}');
    EXECUTE format('ANALYZE public.%I', '${table_name}');
  END IF;
END
\$\$;
SQL

done <<< "$LAYER_NAMES"

echo
echo "Done. Current public tables:"
docker exec -e PGPASSWORD="$PG_PASSWORD" "$PG_CONTAINER" \
  psql -U "$PG_USER" -d "$PG_DB" -c "\dt public.*"