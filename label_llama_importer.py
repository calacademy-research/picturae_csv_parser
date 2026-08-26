import argparse
import ast
import math
import os
import logging
import re

import numpy as np
import pandas as pd
from coordinate_parser.parser import parse_coordinate
from string_utils import remove_non_numerics, detect_is_empty
from BOT_database_updater import UpdateBotDbFields

class ImportLlama:
    def __init__(self, csv_path: str, hemisphere: str = "NorthWest"):
        self.csv_path = csv_path
        self.hemisphere = hemisphere

        self.record_full = pd.read_csv(
            csv_path,
            dtype=str,
            low_memory=False,
        )

        self.clean_llamaframe()

    def remove_artifacts(self):
        """
        Clean artifacts from every text column.

        Removes:
            - {...} and enclosed content
            - <...> and enclosed content
            - ##...## and enclosed content
            - cells containing only quotes, backticks, hashes, $, or whitespace
            - placeholder values such as "empty"
        """
        string_columns = self.record_full.select_dtypes(
            include=["object", "string"]
        ).columns

        def clean_value(value):
            if pd.isna(value):
                return pd.NA

            text = str(value)

            # Repeatedly remove nested enclosed artifacts.
            for _ in range(10):
                previous = text

                text = re.sub(r"\{[^{}]*\}", "", text, flags=re.DOTALL)
                text = re.sub(r"<[^<>]*>", "", text, flags=re.DOTALL)
                text = re.sub(r"##.*?##", "", text, flags=re.DOTALL)

                if text == previous:
                    break

            # Remove unmatched enclosure characters and normalize whitespace.
            text = re.sub(r"[<>{}]", "", text)
            text = re.sub(r"\s+", " ", text).strip()

            # Remove cells containing only artifact characters.
            meaningful_text = re.sub(
                r"""["'`“”‘’#$\s]""",
                "",
                text,
            )

            if not meaningful_text:
                return pd.NA

            if meaningful_text.lower() in {
                "empty",
                "none",
                "null",
                "nan",
                "unknown",
            }:
                return pd.NA

            return text

        self.record_full[string_columns] = self.record_full[
            string_columns
        ].apply(lambda column: column.map(clean_value))




    def parse_list_value(self, value):
        """
        Convert a CSV cell into a Python list.

        Supports:
            [450.0, 1500.0]
            ['m', 'ft']
            [m, ft]
            {450.0, 1500.0}
            {[450.0, 1500.0]}
            m
        """
        if detect_is_empty(value):
            return []

        value_string = str(value).strip()

        try:
            parsed = ast.literal_eval(value_string)

            if isinstance(parsed, list):
                return parsed

            if isinstance(parsed, tuple):
                return list(parsed)

            if isinstance(parsed, set):
                return list(parsed)

            return [parsed]

        except (ValueError, SyntaxError, TypeError):
            # Remove outer list, tuple, or set characters.
            cleaned = value_string.strip().strip("[](){}")

            # Handle an extra nested wrapper such as {[450, 1500]}.
            cleaned = cleaned.strip().strip("[](){}")

            return [
                item.strip().strip("'\"")
                for item in cleaned.split(",")
                if item.strip()
            ]

    def get_verbatim_elevation_pairs(self, verbatim_elevation):
        """extracts numeric elevation value from sets"""
        if detect_is_empty(verbatim_elevation):
            return []

        text = str(verbatim_elevation)

        pairs = []

        number_pattern = r"\d[\d,]*(?:\.\d+)?"
        unit_pattern = r"(?:ft\.?|feet|foot|'|m\.?|meters?|metres?|dm)"

        # Ranges where one unit applies to both values.
        range_matches = re.findall(
            rf"""
                (?P<minimum>{number_pattern})
                \s*[-–—]\s*
                (?P<maximum>{number_pattern})
                \s*
                (?P<unit>{unit_pattern})
                """,
            text,
            flags=re.IGNORECASE | re.VERBOSE,
        )

        for minimum, maximum, unit in range_matches:
            normalized_unit = self.parse_elevation_unit(unit)

            if normalized_unit in {"m", "ft", "dm"}:
                pairs.append((
                    float(minimum.replace(",", "")),
                    normalized_unit
                ))
                pairs.append((
                    float(maximum.replace(",", "")),
                    normalized_unit
                ))

        # Individual elevation values.
        single_matches = re.findall(
            rf"""
                (?P<value>{number_pattern})
                \s*
                (?P<unit>{unit_pattern})
                """,
            text,
            flags=re.IGNORECASE | re.VERBOSE,
        )

        for value, unit in single_matches:
            normalized_unit = self.parse_elevation_unit(unit)

            pair = (
                float(value.replace(",", "")),
                normalized_unit
            )

            if (
                    normalized_unit in {"m", "ft", "dm"}
                    and pair not in pairs
            ):
                pairs.append(pair)

        return pairs


    def filter_pairs_to_verbatim(
            self,
            values,
            units,
            verbatim_elevation,
    ):
        allowed_pairs = self.get_verbatim_elevation_pairs(
            verbatim_elevation
        )

        if not allowed_pairs:
            return []

        filtered_pairs = []

        for value, unit in zip(values, units):
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue

            normalized_unit = self.parse_elevation_unit(unit)

            pair = (numeric_value, normalized_unit)

            if pair in allowed_pairs:
                filtered_pairs.append(pair)

        return filtered_pairs


    def parse_sea_level(self, verbatim_elevation):
        """
        if verbatim elevation contains sea level, set min elevation to 0
        """
        if detect_is_empty(verbatim_elevation):
            return False

        return (
                re.search(
                    r"\bsea\s+level\b",
                    str(verbatim_elevation),
                    flags=re.IGNORECASE,
                )
                is not None
        )

    def parse_elevation_data(self, elevation_values, elevation_units, verbatim_elevation):
        """extract elevation values and ranges,
             with meters taking priority over equivalent elevations in feet
         """
        if self.parse_sea_level(verbatim_elevation):
            return pd.Series([0, pd.NA, pd.NA])

        values = self.parse_list_value(elevation_values)
        units = self.parse_list_value(elevation_units)

        if not values or not units:
            return pd.Series([pd.NA, pd.NA, pd.NA])

        # If one unit applies to multiple extracted values, use that unit
        if len(units) == 1 and len(values) > 1:
            units = units * len(values)

        # Any other mismatch is ambiguous.
        elif len(values) != len(units):
            return pd.Series([pd.NA, pd.NA, pd.NA])

        pairs = self.filter_pairs_to_verbatim(
            values,
            units,
            verbatim_elevation,
        )

        if not pairs:
            return pd.Series([pd.NA, pd.NA, pd.NA])

        for value, unit in zip(values, units):
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue

            normalized_unit = self.parse_elevation_unit(unit)

            if normalized_unit not in {"m", "ft"}:
                continue

            pairs.append((numeric_value, normalized_unit))

        if not pairs:
            return pd.Series([pd.NA, pd.NA, pd.NA])

        values_by_unit = {
            "m": [value for value, unit in pairs if unit == "m"],
            "ft": [value for value, unit in pairs if unit == "ft"],
        }

        meter_values = values_by_unit["m"]
        feet_values = values_by_unit["ft"]

        # Select the unit with the most aligned values.
        # Meters win ties.
        if len(meter_values) >= len(feet_values):
            selected_values = meter_values
            selected_unit = "m"
        else:
            selected_values = feet_values
            selected_unit = "ft"

        if not selected_values:
            return pd.Series([pd.NA, pd.NA, pd.NA])

        elevation_min = min(selected_values)

        elevation_max = (
            max(selected_values)
            if len(selected_values) > 1
            else pd.NA
        )

        if (
                pd.notna(elevation_max)
                and elevation_min == elevation_max
        ):
            elevation_max = pd.NA

        elevation_min, elevation_max, selected_unit = (
            self.remove_plant_height_elev(
                elevation_min,
                elevation_max,
                selected_unit,
                verbatim_elevation,
            )
        )

        return pd.Series([
            elevation_min,
            elevation_max,
            selected_unit,
        ])

    def parse_elevation_unit(self, value):
        """
        Standardize elevation units.

        Priority:
            1. p.s.m -> ft
            2. ft / feet / foot
            3. m / meter / meters
            4. dm

        This ensures values containing both "dm" and "ft" or "m"
        retain the usable ft/m unit.
        """
        if detect_is_empty(value):
            return pd.NA

        unit = str(value).lower().strip()

        # meters take priority
        if re.search(r"\b(?:m|meter|meters|metre|metres)\b", unit):
            return "m"

        # p.s.m as feet
        if re.search(r"\bp\.?\s*s\.?\s*m\.?\b", unit):
            return "ft"

        if re.search(r"(?:\bft\b|\bfoot\b|\bfeet\b|')", unit):
            return "ft"

        # Only return dm when no usable ft or m unit was found.
        if re.search(r"\bdm\b", unit):
            return "dm"

        return pd.NA


    def remove_dm_elevations(self, elevation_min, elevation_max, elevation_unit):
        """
        Empty the parsed elevation fields for rows whose only detected
        elevation unit is dm.

        Rows containing dm together with ft or m are preserved because
        parse_elevation_unit() prioritizes ft and m.
        """
        if elevation_unit == "dm":
            return pd.NA, pd.NA, pd.NA

        return elevation_min, elevation_max, elevation_unit


    def remove_plant_height_elev(self, elevation_min,elevation_max, elevation_unit, verbatim_elevation,):
        """
        Clear single elevation values when VerbatimElevation suggests
        the number describes plant height rather than geographic elevation.
        """

        if detect_is_empty(verbatim_elevation):
            return elevation_min, elevation_max, elevation_unit

        text = str(verbatim_elevation)

        plant_height_pattern = re.compile(
            r"""
            \b
            \d+(?:\.\d+)?          # first number
            \s*[-–—]\s*            # hyphen/en-dash/em-dash
            \d+(?:\.\d+)?          # second number
            \s*
            (?:ft|feet|foot|m|meter|meters|metre|metres|dm)?
            \.?
            \s*
            (?:tall|high|height)
            \b
            """,
            flags=re.IGNORECASE | re.VERBOSE,
        )

        single_height_pattern = re.compile(
            r"""
            \b
            \d+(?:\.\d+)?
            \s*
            (?:ft|feet|foot|m|meter|meters|metre|metres|dm)?
            \.?
            \s*
            (?:tall|high|height)
            \b
            """,
            flags=re.IGNORECASE | re.VERBOSE,
        )

        if plant_height_pattern.search(text) or single_height_pattern.search(text):
            return pd.NA, pd.NA, pd.NA

        return elevation_min, elevation_max, elevation_unit

    def safe_parse_coordinate(self, coordinate, coordinate_type):
        """
        Convert one verbatim coordinate to decimal degrees.

        coordinate_type must be either "latitude" or "longitude".

        Explicit N/S/E/W values take precedence over the default
        hemisphere.
        """
        if detect_is_empty(coordinate):
            return math.nan

        try:
            value = parse_coordinate(str(coordinate), coord_type=coordinate_type)

            if value is None:
                return math.nan

            value = float(value)
            verbatim = str(coordinate).strip().upper()

            hemisphere_defaults = {
                "NorthWest": ("N", "W"),
                "NorthEast": ("N", "E"),
                "SouthWest": ("S", "W"),
                "SouthEast": ("S", "E"),
            }

            latitude_default, longitude_default = (hemisphere_defaults.get(self.hemisphere, ("N", "W")))

            if coordinate_type.lower() in {"lat", "latitude"}:
                if "S" in verbatim:
                    return -abs(value)

                if "N" in verbatim:
                    return abs(value)

                if latitude_default == "N":
                    return abs(value)

                return -abs(value)

            if coordinate_type.lower() in {"lon", "long", "longitude"}:
                if "W" in verbatim:
                    return -abs(value)

                if "E" in verbatim:
                    return abs(value)

                if longitude_default == "W":
                    return -abs(value)

                return abs(value)

            return math.nan

        except (TypeError, ValueError, AttributeError):
            return math.nan

    def coordinate_conversion_failed(self, row) -> bool:
        """
        Return True when a supplied verbatim coordinate could not
        be converted.

        A row is also flagged when only one member of the coordinate
        pair is present.
        """
        verbatim_latitude_present = not detect_is_empty(row.get("verbatimLatitude"))

        verbatim_longitude_present = not detect_is_empty(row.get("verbatimLongitude"))

        latitude_converted = pd.notna(row.get("latitude"))
        longitude_converted = pd.notna(row.get("longitude"))

        # No coordinate values were supplied.
        if not verbatim_latitude_present and not verbatim_longitude_present:
            return False

        # One side of the coordinate pair is missing.
        if verbatim_latitude_present != verbatim_longitude_present:
            return True

        # Both verbatim values are present, but one or both failed.
        return not (latitude_converted and longitude_converted)

    def get_lat_long_unit(self, row):
        """
        Determine the original latitude/longitude format.

        Returns:
            0 = Decimal degrees
            1 = Degrees/minutes/seconds (DMS)
            2 = Degrees/decimal minutes (DM)
        """
        lat = ("" if detect_is_empty(row.get("verbatimLatitude")) else str(row.get("verbatimLatitude")).strip())

        lon = ("" if detect_is_empty(row.get("verbatimLongitude")) else str(row.get("verbatimLongitude")).strip())

        coordinate_text = f"{lat} {lon}"

        # Normalize Unicode prime symbols.
        coordinate_text = (
            coordinate_text
            .replace("′", "'")
            .replace("’", "'")
            .replace("″", '"')
            .replace("“", '"')
            .replace("”", '"')
        )

        # DMS:
        dms_patterns = [
            r"""
            \d+(?:\.\d+)?       # degrees
            \s*°?\s*
            \d+(?:\.\d+)?       # minutes
            \s*[':]\s*
            \d+(?:\.\d+)?       # seconds
            \s*"?
            """,

            r"""
            \d+(?:\.\d+)?
            \s*:\s*
            \d+(?:\.\d+)?
            \s*:\s*
            \d+(?:\.\d+)?
            """,
        ]

        if any(
                re.search(pattern, coordinate_text, flags=re.VERBOSE)
                for pattern in dms_patterns
        ):
            return 1

        # DM:
        dm_patterns = [
            r"""
            \d+(?:\.\d+)?       # degrees
            \s*°?\s*
            \d+(?:\.\d+)?       # minutes
            \s*'
            """,

            r"""
            \d+(?:\.\d+)?
            \s*:\s*
            \d+(?:\.\d+)?
            """,
        ]

        if any(
                re.search(pattern, coordinate_text, flags=re.VERBOSE)
                for pattern in dm_patterns
        ):
            return 2

        # Otherwise assume decimal degrees.
        return 0

    def clean_coordinates(self):
        """
        Convert the single VerbatimLatitude and VerbatimLongitude
        pair into numeric latitude and longitude columns.
        """
        required_columns = {"verbatimLatitude", "verbatimLongitude"}

        missing_columns = required_columns.difference(self.record_full.columns)

        if missing_columns:
            raise KeyError(
                "Missing required coordinate columns: "
                f"{sorted(missing_columns)}"
            )

        self.record_full["lat_long_unit"] = self.record_full.apply(
            self.get_lat_long_unit,
            axis=1,
        )


        self.record_full["latitude"] = self.record_full["verbatimLatitude"].apply(
            lambda value: self.safe_parse_coordinate(value, coordinate_type="latitude"))

        self.record_full["longitude"] = self.record_full["verbatimLongitude"].apply(
            lambda value: self.safe_parse_coordinate(value, coordinate_type="longitude"))

        self.record_full["failed_coordinate_conversion"] = self.record_full.apply(
            self.coordinate_conversion_failed, axis=1)

    def parse_columns(self):
        """combines and extracts information from columns before dropping all but required columns for update."""

        # extracting barcode
        self.record_full["barcode"] = self.record_full["source"].apply(
            lambda x: os.path.splitext(os.path.basename(x))[0]
        )

        # concatenating associated species to habitat.
        self.record_full["habitat"] = (
            self.record_full[["habitat", "associatedTaxa"]]
            .fillna("")
            .astype(str)
            .apply(
                lambda row: ". ".join(
                    value.strip()
                    for value in row
                    if value.strip()
                ),
                axis=1,
            )
        )

        # dropping uneeded columns
        self.record_full.drop(columns=["status", "source", "text", "elapsed", "verbatimEventDate",
                                       "recordedBy", "recordNumber", "identifiedBy", "dateIdentified",
                                       "_elevationValues", "elevationEstimated", "ERROR", "county"], inplace=True)


        # renaming columns to updater standard
        self.record_full.rename(
            columns={
                # Habitat / locality
                "habitat": "Habitat",
                "locality": "LocalityName",

                # lat/long Coordinates
                "latitude": "Latitude1",
                "longitude": "Longitude1",
                "verbatimLatitude": "Lat1Text",
                "verbatimLongitude": "Long1Text",
                # TRS
                "trsTownship": "Township",
                "trsRange": "Range",
                "trsSection": "Section",
                "trsQuad": "BaseMeridian",
                # utm
                "utmNorthing": "UtmNorthing",
                "utmEasting": "UtmEasting",
                "utmZone": "UtmZone",
                # Elevation
                "elevation_min": "MinElevation",
                "elevation_max": "MaxElevation",
                "elevation_unit": "OriginalElevationUnit",
                # Datum, if these are the Llama column names
                "geodeticDatum": "Datum",
            },
            inplace=True,
        )

        self.record_full['LatLongType'] = "Point"
        self.record_full['LatLongMethod'] = "Specimen coord."
        self.record_full["UtmDatum"] = ""

        final_columns = [
            "barcode",
            "country",
            "stateProvince",
            "LocalityName",
            "Habitat",
            "associatedTaxa",
            "occurrenceRemarks",
            "trs",
            "Township",
            "Range",
            "Section",
            "BaseMeridian",
            "utm",
            "UtmNorthing",
            "UtmEasting",
            "UtmZone",
            "UtmDatum",
            "Lat1Text",
            "Long1Text",
            "Latitude1",
            "Longitude1",
            "failed_coordinate_conversion",
            "lat_long_unit",
            "LatLongType",
            "LatLongMethod",
            "verbatimElevation",
            "MinElevation",
            "MaxElevation",
            "OriginalElevationUnit",
        ]

        self.record_full = self.record_full[final_columns]


    def clean_llamaframe(self):

        # remove bracketed llm artifacts.
        self.remove_artifacts()


        # Split elevation values.
        self.record_full[["elevation_min", "elevation_max", "elevation_unit"]] = self.record_full.apply(
                        lambda row: self.parse_elevation_data(row["_elevationValues"], row["elevationUnits"],
                                                              row["verbatimElevation"]), axis=1)

        #standardize empty cells:
        self.record_full = self.record_full.replace(r"(?i)^\s*(nan|none|null|unknown|unkown|empty|<na>|\(empty\)|"
                                                    r"\(empty string\))\s*$",
                                                    pd.NA, regex=True)

        # Convert the single latitude/longitude pair.
        self.clean_coordinates()

        output_directory = os.path.join("update_csv", "update_csv_test")

        os.makedirs(output_directory, exist_ok=True)

        input_basename = os.path.splitext(os.path.basename(self.csv_path))[0]

        output_path = os.path.join(output_directory, f"{input_basename}_output.csv")

        self.parse_columns()

        self.record_full.to_csv(output_path, index=False)

        logging.info(f"Cleaned CSV written to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=("Import Llama CSV results and clean elevation "
                     "and coordinate fields."))

    parser.add_argument("-c", "--csv-path",
                        required=True, help="Path to the input CSV file.")

    parser.add_argument("-hm", "--hemisphere", default="NorthWest",
                        choices=["NorthWest", "NorthEast", "SouthWest", "SouthEast"],
                        help=("Default hemisphere for coordinates without explicit "
                               "N, S, E, or W indicators. Default: NorthWest."))

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    importer = ImportLlama(
        csv_path=args.csv_path,
        hemisphere=args.hemisphere)