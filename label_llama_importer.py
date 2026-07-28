import argparse
import ast
import math
import os
import logging
import re

import pandas as pd
from coordinate_parser.parser import parse_coordinate
from string_utils import remove_non_numerics
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

    def detect_is_empty(self, value) -> bool:
        """
        Detect empty or none-like values.
        """
        if value is None:
            return True

        try:
            if isinstance(value, float) and math.isnan(value):
                return True
        except (TypeError, ValueError):
            pass

        return str(value).strip().lower() in {"", "nan", "none", "null", "unknown", "unkown"}

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
        if self.detect_is_empty(value):
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

    def parse_elevation_data(self, elevation_values, elevation_units):
        """
        Pair elevations with units and select the most likely elevation range.

        Rules:
            - One unit applies to every value, e.g. 200-300 ft.
            - Ignore dm and unknown units.
            - Select the unit with the most values.
            - Meters win when meter and feet counts are tied.
        """
        values = self.parse_list_value(elevation_values)
        units = self.parse_list_value(elevation_units)

        if not values or not units:
            return pd.Series([pd.NA, pd.NA, pd.NA])

        # Example:
        # values = [200, 300]
        # units  = ["ft"]
        # becomes ["ft", "ft"]
        if len(units) == 1 and len(values) > 1:
            units = units * len(values)

        elevations_by_unit = {
            "m": [],
            "ft": [],
        }

        for value, unit in zip(values, units):
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue

            normalized_unit = self.parse_elevation_unit(unit)

            # Ignore dm and unknown units.
            if normalized_unit not in elevations_by_unit:
                continue

            elevations_by_unit[normalized_unit].append(numeric_value)

        meter_count = len(elevations_by_unit["m"])
        feet_count = len(elevations_by_unit["ft"])

        if meter_count == 0 and feet_count == 0:
            return pd.Series([pd.NA, pd.NA, pd.NA])

        # Most frequently occurring unit wins.
        # Meters win ties.
        selected_unit = "m" if meter_count >= feet_count else "ft"
        selected_values = elevations_by_unit[selected_unit]

        elevation_min = min(selected_values)

        elevation_max = (
            max(selected_values)
            if len(selected_values) > 1
            else pd.NA
        )

        if pd.notna(elevation_max) and elevation_min == elevation_max:
            elevation_max = pd.NA

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
        if self.detect_is_empty(value):
            return pd.NA

        unit = str(value).lower().strip()

        # meters take priority
        if re.search(r"\b(?:m|meter|meters|metre|metres)\b", unit):
            return "m"

        # p.s.m as feet
        if re.search(r"\bp\.?\s*s\.?\s*m\.?\b", unit):
            return "ft"

        if re.search(r"\b(?:ft|foot|feet)\b", unit):
            return "ft"

        # Only return dm when no usable ft or m unit was found.
        if re.search(r"\bdm\b", unit):
            return "dm"

        return pd.NA


    def remove_dm_elevations(self):
        """
        Empty the parsed elevation fields for rows whose only detected
        elevation unit is dm.

        Rows containing dm together with ft or m are preserved because
        parse_elevation_unit() prioritizes ft and m.
        """
        dm_mask = self.record_full["elevation_unit"].eq("dm")

        columns_to_empty = [
            "elevation_min",
            "elevation_max",
            "elevation_unit",
        ]

        self.record_full.loc[dm_mask, columns_to_empty] = pd.NA

        logging.info(
            "Removed elevation data from %s dm-only rows.",
            int(dm_mask.sum()),
        )

    def safe_parse_coordinate(self, coordinate, coordinate_type):
        """
        Convert one verbatim coordinate to decimal degrees.

        coordinate_type must be either "latitude" or "longitude".

        Explicit N/S/E/W values take precedence over the default
        hemisphere.
        """
        if self.detect_is_empty(coordinate):
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
        verbatim_latitude_present = not self.detect_is_empty(row.get("verbatimLatitude"))

        verbatim_longitude_present = not self.detect_is_empty(row.get("verbatimLongitude"))

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

        self.record_full["latitude"] = self.record_full["verbatimLatitude"].apply(
            lambda value: self.safe_parse_coordinate(value, coordinate_type="latitude"))

        self.record_full["longitude"] = self.record_full["verbatimLongitude"].apply(
            lambda value: self.safe_parse_coordinate(value, coordinate_type="longitude"))

        self.record_full["failed_coordinate_conversion"] = self.record_full.apply(
            self.coordinate_conversion_failed, axis=1)

    def clean_llamaframe(self):
        # Split elevation values.
        self.record_full[["elevation_min", "elevation_max", "elevation_unit"]] = self.record_full.apply(
                        lambda row: self.parse_elevation_data(row["_elevationValues"], row["elevationUnits"]), axis=1)

        # removing plant height measurements from elevation
        self.remove_dm_elevations()

        # Convert the single latitude/longitude pair.
        self.clean_coordinates()

        output_directory = os.path.join("update_csv", "update_csv_test")

        os.makedirs(output_directory, exist_ok=True)

        input_basename = os.path.splitext(os.path.basename(self.csv_path))[0]

        output_path = os.path.join(output_directory,f"{input_basename}_output.csv")

        self.record_full.to_csv(output_path,index=False)

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