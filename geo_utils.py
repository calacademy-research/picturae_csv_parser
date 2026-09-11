import math
import pandas as pd
from pyproj import Transformer
import re
from string_utils import detect_is_empty

def euclidean_m(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def append_remark(existing, new_remark):
    existing = "" if pd.isna(existing) else str(existing).strip()

    if not existing or existing.lower() == "nan":
        return new_remark

    if new_remark.lower() in existing.lower():
        return existing

    return f"{existing}; {new_remark}"


def test_projected_crs_from_latlong(
    row,
    *,
    test_crs,
    datum_label,
    remark_label,
    threshold_m=5500,
    easting_col="easting",
    northing_col="northing",
    lat_col="latitude_numeric",
    lon_col="longitude_numeric",
):
    """
    Converts existing lat/lon into test_crs and compares the result to
    existing easting/northing.

    If close enough, returns datum/remark labels.
    """

    easting = pd.to_numeric(row.get(easting_col), errors="coerce")
    northing = pd.to_numeric(row.get(northing_col), errors="coerce")
    lat = pd.to_numeric(row.get(lat_col), errors="coerce")
    lon = pd.to_numeric(row.get(lon_col), errors="coerce")

    if pd.isna(easting) or pd.isna(northing) or pd.isna(lat) or pd.isna(lon):
        return None

    transformer = Transformer.from_crs(
        "EPSG:4326",
        test_crs,
        always_xy=True,
    )

    expected_easting, expected_northing = transformer.transform(lon, lat)

    distance_m = euclidean_m(
        easting,
        northing,
        expected_easting,
        expected_northing,
    )

    if distance_m <= threshold_m:
        return {
            "utm_datum": datum_label,
            "locality_det_remarks": remark_label,
            "distance_m": round(distance_m, 2),
            "matched_crs": test_crs,
        }

    return None


def exceptions(row):
    """
    Manual CRS exception rules.

    Add new country/admin-specific CRS tests here as problems crop up.
    """

    country = str(row.get("Country", "")).strip().lower()

    if country == "taiwan":
        return [
            {
                "test_crs": "EPSG:3826",
                "datum_label": "TWD97 / TM2 zone 121",
                "remark_label": "Taiwan Grid",
                "threshold_m": 5500,
            }
        ]

    return []



def apply_crs_exceptions(
    df,
    *,
    datum_col="utm_datum",
    remarks_col="locality_det_remarks",
    distance_col="crs_exception_distance_m",
    matched_crs_col="crs_exception_matched_crs",
):
    df = df.copy()

    for col in [datum_col, remarks_col, distance_col, matched_crs_col]:
        if col not in df.columns:
            df[col] = ""

    for idx, row in df.iterrows():
        for rule in exceptions(row):
            result = test_projected_crs_from_latlong(row, **rule)

            if result is None:
                continue

            df.loc[idx, datum_col] = result["utm_datum"]
            df.loc[idx, remarks_col] = append_remark(
                df.loc[idx, remarks_col],
                result["locality_det_remarks"],
            )
            df.loc[idx, distance_col] = result["distance_m"]
            df.loc[idx, matched_crs_col] = result["matched_crs"]

            # Stop after first matching exception
            break

    return df


def classify_coordinate(value):
    """For classifying verbatim coordinates as either DMS, DM, or DD
       to fill OriginalLatLongUnit/SrcLatLongUnit fields in database locality table.
    """
    if detect_is_empty(value):
        return None

    text = str(value).strip()

    # Normalize common Unicode coordinate symbols.
    text = (
        text
        .replace("′", "'")
        .replace("’", "'")
        .replace("″", '"')
        .replace("“", '"')
        .replace("”", '"')
    )

    # Remove hemisphere indicators so they don't interfere.
    text = re.sub(
        r"\b(?:N|S|E|W)\b\.?",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()

    # Extract numeric components.
    numbers = re.findall(
        r"\d+(?:\.\d+)?",
        text,
    )

    # Explicit seconds marker -> DMS.
    if '"' in text:
        return 1

    # Three numeric components:
    if len(numbers) >= 3:
        return 1

    # Two numeric components:
    if len(numbers) == 2:
        return 2

    # One numeric component:
    if len(numbers) == 1:
        return 0

    return None


def get_lat_long_unit(verbatim_lat, verbatim_long):
    """
    Determine original latitude/longitude format.

    Returns:
        0 = Decimal degrees
        1 = Degrees/minutes/seconds (DMS)
        2 = Degrees/decimal minutes (DM)
    """

    lat_unit = classify_coordinate(
        verbatim_lat
    )

    lon_unit = classify_coordinate(
        verbatim_long
    )

    units = [
        unit
        for unit in (lat_unit, lon_unit)
        if unit is not None
    ]

    if not units:
        return 0

    # If either coordinate is clearly DMS, preserve DMS.
    if 1 in units:
        return 1

    # Otherwise if either is DM, use DM.
    if 2 in units:
        return 2

    return 0