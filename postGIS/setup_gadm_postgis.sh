#!/usr/bin/env bash
set -euo pipefail

# =========================
# Config (edit these)
# =========================
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
GDB_PATH="${1:-"$SCRIPT_DIR/gadm_410.gdb"}"
PG_CONTAINER="gadm-postgis"
PG_DB="gis"
PG_USER="postgres"
PG_PASSWORD="postgres"
PG_PORT="5432"

# =========================
# Checks
# =========================
if [ ! -d "$GDB_PATH" ]; then
  echo "ERROR: GDB folder not found: $GDB_PATH"
  echo "Defaulted to script directory:"
  echo "  $SCRIPT_DIR/gadm_410.gdb"
  echo "Or pass it explicitly:"
  echo "  Usage: $0 /absolute/path/to/gadm_410.gdb"
  exit 1
fi

echo "Using GDB: $GDB_PATH"

if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker is required"
  exit 1
fi

if ! command -v ogrinfo >/dev/null 2>&1 || ! command -v ogr2ogr >/dev/null 2>&1; then
  echo "ERROR: GDAL tools (ogrinfo / ogr2ogr) are required."
  echo "On macOS (brew): brew install gdal"
  exit 1
fi

echo "Using GDB: $GDB_PATH"

# =========================
# Start PostGIS container
# =========================
if docker ps -a --format '{{.Names}}' | grep -qx "$PG_CONTAINER"; then
  echo "Container $PG_CONTAINER already exists. Starting it..."
  docker start "$PG_CONTAINER" >/dev/null || true
else
  echo "Creating PostGIS container $PG_CONTAINER..."
  docker run -d \
    --name "$PG_CONTAINER" \
    -e POSTGRES_DB="$PG_DB" \
    -e POSTGRES_USER="$PG_USER" \
    -e POSTGRES_PASSWORD="$PG_PASSWORD" \
    -p "$PG_PORT:5432" \
    postgis/postgis:16-3.4
fi

# =========================
# Wait for Postgres
# =========================
echo "Waiting for PostgreSQL to be ready..."
until docker exec "$PG_CONTAINER" pg_isready -U "$PG_USER" -d "$PG_DB" >/dev/null 2>&1; do
  sleep 2
done
echo "PostgreSQL is ready."

# =========================
# Enable PostGIS
# =========================
echo "Ensuring PostGIS extension exists..."
docker exec -e PGPASSWORD="$PG_PASSWORD" "$PG_CONTAINER" \
  psql -U "$PG_USER" -d "$PG_DB" -v ON_ERROR_STOP=1 -c "CREATE EXTENSION IF NOT EXISTS postgis;"

# =========================
# List GDB layers
# =========================
echo "Reading layers from FileGDB..."
LAYER_LINES=$(ogrinfo "$GDB_PATH" 2>/dev/null | grep -E '^[0-9]+: ' || true)

if [ -z "$LAYER_LINES" ]; then
  echo "ERROR: No layers found. Verify GDAL can read this .gdb (OpenFileGDB driver)."
  exit 1
fi

echo "Found layers:"
echo "$LAYER_LINES"

# =========================
# Import each layer
# =========================
echo "Importing layers into PostGIS..."

while IFS= read -r line; do
  # Example line usually looks like:
  #  1: gadm_410_0 (Multi Polygon)
  layer_name=$(echo "$line" | sed -E 's/^[0-9]+: //' | sed -E 's/ \(.+$//' | sed 's/[[:space:]]*$//')

  # Make a safe table name
  table_name=$(echo "$layer_name" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9_]/_/g')

  echo "  -> Importing layer '$layer_name' as table '$table_name'"

  ogr2ogr \
    -f PostgreSQL \
    "PG:host=localhost port=$PG_PORT dbname=$PG_DB user=$PG_USER password=$PG_PASSWORD" \
    "$GDB_PATH" \
    "$layer_name" \
    -nln "$table_name" \
    -overwrite \
    -lco GEOMETRY_NAME=geom \
    -lco FID=gid \
    -nlt PROMOTE_TO_MULTI >/dev/null

  # Add spatial index (if geometry exists)
  docker exec -e PGPASSWORD="$PG_PASSWORD" "$PG_CONTAINER" \
    psql -U "$PG_USER" -d "$PG_DB" -v ON_ERROR_STOP=1 <<SQL >/dev/null
DO \$\$
BEGIN
  IF EXISTS (
    SELECT 1
    FROM information_schema.columns
    WHERE table_name = '$table_name'
      AND column_name = 'geom'
  ) THEN
    EXECUTE 'CREATE INDEX IF NOT EXISTS ${table_name}_geom_gix ON $table_name USING GIST (geom)';
    EXECUTE 'ANALYZE $table_name';
  END IF;
END
\$\$;
SQL

done <<< "$LAYER_LINES"

echo
echo "Done."
echo "PostGIS DB: $PG_DB"
echo "Host: localhost  Port: $PG_PORT"
echo "User: $PG_USER   Password: $PG_PASSWORD"
echo
echo "Quick test:"
echo "  docker exec -e PGPASSWORD=$PG_PASSWORD $PG_CONTAINER psql -U $PG_USER -d $PG_DB -c '\\dt'"