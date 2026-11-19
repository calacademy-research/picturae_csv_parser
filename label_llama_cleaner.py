import argparse
import ast
import math
import os
import logging
import re
import pandas as pd
from get_configs import get_config
from coordinate_parser.parser import parse_coordinate
from string_utils import detect_is_empty
from difflib import SequenceMatcher
from html import unescape
from gen_import_utils import clean_numeric_column, clean_utm_zone, correct_swapped_utm
from geo_utils import get_lat_long_unit
from sql_csv_utils import SqlCsvTools

class LlamaClean:
    # A standalone zero is a placeholder only in descriptive text fields.
    # Keep valid zeros in coordinates, elevations, identifiers, and flags.
    TEXT_COLUMNS = frozenset({
        "country", "stateProvince", "locality", "LocalityName",
        "Habitat", "associatedTaxa", "occurrenceRemarks", "specimen_description",
    })

    def __init__(self, csv_path: str, hemisphere: str = "NorthWest", config="Botany_PIC"):
        self.config = get_config(config)
        self.sql_csv_tools = SqlCsvTools(config=self.config)
        self.csv_path = csv_path
        self.hemisphere = hemisphere

        self.record_full = pd.read_csv(
            csv_path,
            dtype=str,
            low_memory=False,
        )

        self.clean_llamaframe()


    @staticmethod
    def clean_artifact_value(value, *, zero_is_empty=False):
        """Return pd.NA for artifact-only cells; retain punctuation in real text.

        The existing removal of {...}, <...>, and ##...## content is retained.
        zero_is_empty must only be enabled for descriptive text fields.
        """
        if pd.isna(value):
            return pd.NA

        # Decode escaped quotes/tags, nonbreaking spaces, and similar entities.
        text = unescape(str(value))
        text = re.sub(r"[\u200b-\u200d\u2060\ufeff]", "", text)

        # Repeatedly remove nested enclosed artifacts, as in the original cleaner.
        for _ in range(10):
            previous = text
            text = re.sub(r"\{[^{}]*\}", " ", text, flags=re.DOTALL)
            text = re.sub(r"<[^<>]*>", " ", text, flags=re.DOTALL)
            text = re.sub(r"##.*?##", " ", text, flags=re.DOTALL)

            if text == previous:
                break

        text = re.sub(r"[<>{}]", "", text)
        text = re.sub(r"\s+", " ", text).strip()

        # Catch ANY punctuation/symbol-only cell, including >>, '';, ||, and ]].
        # isalnum() also preserves words containing accented/non-Latin letters.
        if not any(character.isalnum() for character in text):
            return pd.NA

        # Ignore outer punctuation only for the placeholder check. Keep the
        # original punctuation when the value contains meaningful text.
        placeholder = re.sub(r"^[\W_]+|[\W_]+$", "", text).casefold()
        if placeholder in {
            "empty", "empty string", "none", "null", "nan", "na", "n/a",
            "n.a", "unknown", "unkown", "not available", "not provided",
            "not recorded", "not specified", "no data",
        }:
            return pd.NA

        if zero_is_empty and re.fullmatch(r"0+(?:\.0+)?", placeholder):
            return pd.NA

        return text

    def clean_text_fields(self):
        """Recheck descriptive fields after renaming and database lookups."""
        for column in self.record_full.columns:
            if column in self.TEXT_COLUMNS:
                self.record_full[column] = self.record_full[column].map(
                    lambda value: self.clean_artifact_value(value, zero_is_empty=True)
                )

    def remove_artifacts(self):
        """Clean string columns before parsing; clear text-only zero placeholders."""
        string_columns = self.record_full.select_dtypes(
            include=["object", "string"]
        ).columns

        for column in string_columns:
            self.record_full[column] = self.record_full[column].map(
                lambda value: self.clean_artifact_value(
                    value, zero_is_empty=column in self.TEXT_COLUMNS,
                )
            )

        for column in ["utmNorthing", "utmEasting"]:
            self.record_full[column] = self.record_full[column].apply(clean_numeric_column)

        self.record_full["utmZone"] = self.record_full["utmZone"].apply(clean_utm_zone)

    def remove_echoed_column_names(self, threshold=0.90):
        """Clear cells that fuzzily match their own column name."""

        def normalize(value):
            if pd.isna(value):
                return ""

            return re.sub(
                r"[^a-z0-9]",
                "",
                str(value).casefold(),
            )

        # Avoid altering metadata required later in the pipeline.
        excluded_columns = {
            "source",
            "status",
            "text",
            "elapsed",
        }

        for column in self.record_full.columns:
            if column in excluded_columns:
                continue

            normalized_column = normalize(column)

            def matches_column_name(value):
                normalized_value = normalize(value)

                if not normalized_value:
                    return False

                # Always remove an exact normalized match.
                if normalized_value == normalized_column:
                    return True

                # Avoid fuzzy matching very short names such as trs or utm.
                if len(normalized_column) < 5:
                    return False

                similarity = SequenceMatcher(
                    None,
                    normalized_value,
                    normalized_column,
                ).ratio()

                return similarity >= threshold

            mask = self.record_full[column].map(matches_column_name)

            if mask.any():
                logging.debug(
                    "Removed echoed column name from column %s in %d row(s)",
                    column,
                    mask.sum(),
                )

                self.record_full.loc[mask, column] = pd.NA


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

            if isinstance(parsed, (tuple, set)):
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

            if pd.isna(normalized_unit):
                continue

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
        empty_result = (pd.NA, pd.NA, pd.NA)

        if self.parse_sea_level(verbatim_elevation):
            return pd.Series([0, pd.NA, pd.NA])

        values = self.parse_list_value(elevation_values)
        units = self.parse_list_value(elevation_units)

        if not values or not units:
            return empty_result

        # If one unit applies to multiple extracted values, use that unit
        if len(units) == 1 and len(values) > 1:
            units = units * len(values)
        elif len(values) != len(units):
            return empty_result

        pairs = self.filter_pairs_to_verbatim(values, units, verbatim_elevation)

        values_by_unit = {"m": [], "ft": []}

        for value, unit in pairs:
            if unit in values_by_unit:
                values_by_unit[unit].append(value)

        meter_values = values_by_unit["m"]
        feet_values = values_by_unit["ft"]

        if len(meter_values) >= len(feet_values):
            selected_values, selected_unit = meter_values, "m"
        else:
            selected_values, selected_unit = feet_values, "ft"

        if not selected_values:
            return empty_result

        elevation_min = min(selected_values)
        elevation_max = max(selected_values)

        if elevation_min == elevation_max:
            elevation_max = pd.NA

        return self.remove_plant_height_elev(
            elevation_min,
            elevation_max,
            selected_unit,
            verbatim_elevation,
        )


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


    def remove_plant_height_elev(self, elevation_min,elevation_max, elevation_unit, verbatim_elevation,):
        """
        Clear single elevation values when VerbatimElevation suggests
        the number describes plant height rather than geographic elevation.
        """

        if not detect_is_empty(verbatim_elevation):
            pattern = r"""
                    \b
                    \d+(?:\.\d+)?
                    \s*
                    (?:ft|feet|foot|m|meter|meters|metre|metres|dm)?
                    \.?
                    \s*
                    (?:tall|high|height)
                    \b
                """

            if re.search(pattern, str(verbatim_elevation), flags=re.IGNORECASE | re.VERBOSE):
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


    def clean_coordinates(self):
        """
        Convert the single VerbatimLatitude and VerbatimLongitude
        pair into numeric latitude and longitude columns.
        """

        required_columns = {"verbatimLatitude", "verbatimLongitude"}
        missing_columns = required_columns.difference(self.record_full.columns)

        if missing_columns:
            raise KeyError(
                "Missing required coordinate columns: {sorted(missing_columns)}"
            )

        self.record_full["OriginalLatLongUnit"] = [get_lat_long_unit(latitude, longitude)
                                              for latitude, longitude in zip(self.record_full["verbatimLatitude"],
                                                                             self.record_full["verbatimLongitude"])]

        self.record_full["SrcLatLongUnit"] = self.record_full["OriginalLatLongUnit"]

        self.record_full["latitude"] = self.record_full["verbatimLatitude"].map(
            lambda value: self.safe_parse_coordinate(value, coordinate_type="latitude"))

        self.record_full["longitude"] = self.record_full["verbatimLongitude"].map(
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
        self.record_full["Habitat"] = [self.combine_habitat_and_taxa(habitat, taxa) for habitat, taxa in zip(
                                       self.record_full["Habitat"], self.record_full["associatedTaxa"])]

        # renaming columns to updater standard
        self.record_full.rename(
            columns={
                # Habitat / locality
                "Habitat": "Habitat",
                "occurrenceRemarks": "specimen_description",
                "locality": "LocalityName",

                # lat/long Coordinates
                "latitude": "Latitude1",
                "longitude": "Longitude1",
                "verbatimLatitude": "Lat1Text",
                "verbatimLongitude": "Long1Text",
                # TRS
                "trsTownship": "Township",
                "trsRange": "RangeDesc",
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

        self.record_full['LatLongMethod'] = "Specimen coord."
        self.record_full[['Datum', 'UtmDatum', "Latitude2", "Longitude2", "Lat2Text", "Long2Text"]] = ""

        final_columns = [
            "barcode",
            "country",
            "stateProvince",
            "LocalityName",
            "Habitat",
            "associatedTaxa",
            "specimen_description",
            "trs",
            "Township",
            "RangeDesc",
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
            "Lat2Text",
            "Long2Text",
            "Latitude2",
            "Longitude2",
            "failed_coordinate_conversion",
            "OriginalLatLongUnit",
            "SrcLatLongUnit",
            "LatLongMethod",
            "Datum",
            "verbatimElevation",
            "MinElevation",
            "MaxElevation",
            "OriginalElevationUnit",
        ]

        self.record_full = self.record_full[final_columns]

    def combine_habitat_and_taxa(self, habitat, associated_taxa):
        """Combine habitat and taxa, excluding record dumps and duplicates."""

        def clean(value):
            if detect_is_empty(value):
                return ""

            text = re.sub(r"\s+", " ", str(value)).strip()

            # Treat punctuation-only values such as "" as empty.
            if not any(character.isalnum() for character in text):
                return ""

            # Reject the record-dump pattern shown in your examples:
            # four or more consecutive empty quoted fields.
            if re.search(r'(?:^|\s)(?:""\s+){3,}""(?=\s|$)', text):
                return ""

            return text

        habitat = clean(habitat)
        associated_taxa = clean(associated_taxa)

        if not habitat:
            return associated_taxa

        if not associated_taxa:
            return habitat

        def normalize(text):
            return re.sub(r"[.,;:\s]+$", "", text.casefold()).strip()

        habitat_norm = normalize(habitat)
        taxa_norm = normalize(associated_taxa)

        if taxa_norm in habitat_norm:
            return habitat

        if habitat_norm in taxa_norm:
            return associated_taxa

        similarity = SequenceMatcher(
            None, habitat_norm, taxa_norm
        ).ratio()

        if similarity >= 0.90:
            return max((habitat, associated_taxa), key=len)

        return f"{habitat.rstrip('.')}. {associated_taxa}"

    def replace_locality_names_from_database(self):
        """Replace the output LocalityName column with database values."""
        query = """
            SELECT l.LocalityName
            FROM collectionobject AS co
            JOIN collectingevent AS ce
                ON ce.CollectingEventID = co.CollectingEventID
            JOIN locality AS l
                ON l.LocalityID = ce.LocalityID
            WHERE co.CatalogNumber = %s
        """

        barcodes = self.record_full["barcode"].astype("string").str.strip()
        locality_names = {}

        # Look up each barcode once, preserving leading zeros.
        for barcode in barcodes.dropna().unique():
            if not barcode:
                continue

            logging.info(f"fetching locality for barcode: {barcode}")
            rows = self.sql_csv_tools.get_records(
                sql=query,
                params=(str(barcode).zfill(9),),
            )

            if not rows:
                logging.warning(
                    "No linked database locality found for barcode %s",
                    barcode,
                )
                continue

            if len(rows) > 1:
                raise ValueError(
                    f"Multiple database records match CatalogNumber "
                    f"{barcode!r}; cannot choose a locality."
                )

            locality_names[barcode] = rows[0][0]

        self.record_full["LocalityName"] = (
            barcodes.map(locality_names).astype("string")
        )


    def compare_locality_and_habitat(self):
        """Compare shared phrases containing at least two consecutive words."""

        def normalize(value):
            if pd.isna(value):
                return []

            # Ignore case and punctuation while preserving word boundaries.
            return re.findall(r"[^\W_]+", str(value).casefold())

        def calculate_overlap(locality_words, habitat_words):
            if not locality_words or not habitat_words:
                return 0.0

            matcher = SequenceMatcher(
                None,
                locality_words,
                habitat_words,
                autojunk=False,
            )

            # Count only matching phrases of two or more consecutive words.
            matching_words = sum(
                block.size
                for block in matcher.get_matching_blocks()
                if block.size >= 2
            )

            return (
                2.0 * matching_words
                / (len(locality_words) + len(habitat_words))
            )

        localities = self.record_full["LocalityName"].map(normalize)
        habitats = self.record_full["Habitat"].map(normalize)

        self.record_full["locality_habitat_overlap"] = [
            calculate_overlap(locality, habitat)
            for locality, habitat in zip(localities, habitats)
        ]

        self.record_full["locality_habitat_overlap_gt_20pct"] = (
            self.record_full["locality_habitat_overlap"] > 0.20
        )

    def clean_llamaframe(self):

        # remove llm artifacts.
        self.remove_artifacts()
        self.remove_echoed_column_names(threshold=0.90)


        # Split elevation values.

        elevation_cols = ["elevation_min", "elevation_max", "elevation_unit"]

        self.record_full[elevation_cols] = pd.DataFrame([self.parse_elevation_data(values, units, verbatim)
                                                         for values, units, verbatim in zip(
                                                         self.record_full["_elevationValues"],
                                                         self.record_full["elevationUnits"],
                                                         self.record_full["verbatimElevation"],)],
                                                        index=self.record_full.index, columns=elevation_cols)


        #standardize empty cells:
        self.record_full = self.record_full.replace(
            r"(?i)^\s*(?:nan|none|null|unknown|unkown|empty|<na>|\[empty\]|\(empty\)|\(empty string\)|!!|\?\?|��|॥)\s*$",
            pd.NA,
            regex=True,
        )


        # Correct swapped UTM northing/easting values.
        utm_cols = ["utmNorthing", "utmEasting"]

        self.record_full[utm_cols] = pd.DataFrame([correct_swapped_utm(northing, easting) for northing, easting in zip(
                                                  self.record_full["utmNorthing"],
                                                  self.record_full["utmEasting"])],
                                                  index=self.record_full.index,
                                                  columns=utm_cols)

        # Convert the single latitude/longitude pair.
        self.clean_coordinates()

        output_directory = os.path.join("update_csv", "update_csv_test")

        os.makedirs(output_directory, exist_ok=True)

        input_basename = os.path.splitext(os.path.basename(self.csv_path))[0]

        output_path = os.path.join(output_directory, f"{input_basename}_output.csv")

        self.parse_columns()

        self.replace_locality_names_from_database()

        # Also clean renamed/combined fields and values returned by the database.
        self.clean_text_fields()

        self.compare_locality_and_habitat()

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
                        help="Default hemisphere for coordinates without explicit "
                             "N, S, E, or W indicators. Default: NorthWest.")

    parser.add_argument("-cfg", "--config", default="Botany_PIC", required=False,
                        help="config file to retrieve for sql")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    importer = LlamaClean(csv_path=args.csv_path, hemisphere=args.hemisphere, config=args.config)
