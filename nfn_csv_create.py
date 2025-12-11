import pandas as pd
import time_utils
from collections import defaultdict
import os
from get_configs import get_config
from sql_csv_utils import SqlCsvTools
from string_utils import remove_non_numerics
import numpy as np
import csv
import logging
import json5
import json
import requests
import argparse
import math
from label_reconciliations.core import run_on_dataframe
from coordinate_parser.parser import parse_coordinate

# https://pypi.org/project/coordinate-parser/

class NfnCsvCreate():
    def __init__(self, coll, input_file, logging_level, hemisphere):

        self.logger = logging.getLogger("NfnCreatePicturae")
        self.logger.setLevel(logging_level)

        self.config = get_config(coll)

        self.input_file = input_file

        self.row = None
        self.index = None

        self.master_csv = self.read_and_concat_csvs()

        self.unpack_json()

        self.datums = self.config.ACCEPTED_DATUMS

        self.sql_csv_tools = SqlCsvTools(config=self.config)

        self.ollama_url = self.config.OLLAMA_URL

        self.hemisphere = hemisphere

        self.nominatum_dict = {}

        # writing out uncleaned df for testing purposes
        self.master_csv.to_csv(
            f"nfn_csv{os.path.sep}nfn_csv_output{os.path.sep}uncleaned_nfn_{time_utils.get_pst_time_now()}.csv",
            sep=',',
            quoting=csv.QUOTE_ALL,
            index=False
        )


    def read_in_prompt(self, filepath):
        """reads in prompts from .txt files"""
        with open(f"{filepath}", "r", encoding="utf-8") as f:
            prompt = f.read()
        return prompt

    def read_and_concat_csvs(self):
        """reads in and concatenates all csvs in the nfn csv folder"""
        csv_folder = self.config.NFN_CSV_FOLDER
        if not os.path.exists(csv_folder):
            raise FileNotFoundError(f"The folder {csv_folder} does not exist.")

        csv_file = os.path.join(csv_folder, self.input_file)

        self.logger.info(f"Found {self.input_file} CSV in {csv_folder}")

        master_csv = pd.read_csv(csv_file, dtype=str, low_memory=False)

        return master_csv

    def parse_json_cell(self, cell):
        """Splits JSON into columns, numbering only coordinate-related fields."""
        items = json.loads(cell)
        output = {}
        counts = defaultdict(int)

        coordinate_keywords = ["coordinate", "latitude", "longitude", "utm", "township", "range", "section",
                               "datum", "quadrangle"]

        for it in items:
            label = (it.get("task_label") or it.get("task") or "").strip()
            val = it.get("value", None)

            if any(kw in label.lower() for kw in coordinate_keywords):
                counts[label] += 1
                key = f"{label}_{counts[label]}"
            else:
                key = label

            output[key] = val

        return output

    def extract_value(self, cell, value):
        """extracts a value from a json"""
        d = json.loads(cell)
        inner = next(iter(d.values()))
        return inner.get(value, None)


    def detect_is_empty(self, string) -> bool:
        """method to detect if a string type variable contains none or none-like value.
           Returns True and False
        """
        if string is None:
            return True
        try:
            if isinstance(string, float) and math.isnan(string):
                return True
        except Exception:
            pass
        s = str(string).strip().lower()
        return s in ("", "nan", "none", "null", 'unknown', 'unkown')

    def unpack_json(self):
        """function used to unpack json blobs in standard nfn classification ouput"""

        self.master_csv.drop(columns=["metadata"], inplace=True)

        # expanding out the annotation fields into columns
        expanded = self.master_csv["annotations"].apply(self.parse_json_cell)

        expanded_df = pd.DataFrame(expanded.tolist())

        self.master_csv = pd.concat([self.master_csv.drop(columns=["annotations"]), expanded_df], axis=1)

        # extracting barcode field
        for field_name in ["Barcode", "Country", "State", "County", "CollectorNumber"]:
            self.master_csv[field_name] = self.master_csv["subject_data"].apply(lambda cell: self.extract_value(cell, field_name))

        self.master_csv.drop(columns=["subject_data", "user_ip", "created_at", "gold_standard", "expert"], inplace=True)

        # moving barcode to front of dataframe
        cols = ["Barcode", "Country", "State", "County", "CollectorNumber"] + [col for col in self.master_csv.columns if col not in
                                                            ["Barcode", "Country", "State", "County"]]

        self.master_csv = self.master_csv[cols]



    def rename_columns(self):

        col_dict = self.config.NFN_COL_DICT
        # Filter the dictionary to only include columns present in the original master_csv
        existing_columns = self.master_csv.columns
        filtered_col_dict = {k: v for k, v in col_dict.items() if k in existing_columns}

        # Rename columns using the filtered dictionary
        self.master_csv = self.master_csv.rename(columns=filtered_col_dict)

        # Start with the known columns
        known_cols = list(filtered_col_dict.values())

        # Add the remaining columns at the end (preserve extras)
        remaining_cols = [col for col in self.master_csv.columns if col not in known_cols]

        final_col_order = known_cols + remaining_cols

        self.master_csv = self.master_csv[final_col_order]

    def remove_records(self):
        """remove_records: removes potential problem records before cleaning.
          Includes double transcriptions, banned user names."""
        # Removing blacklist records
        blacklist = self.config.NFN_BLACKLIST

        barcode_set = set(self.master_csv['Barcode'].unique())

        self.master_csv = self.master_csv[~self.master_csv['user_name'].isin(blacklist)].reset_index(drop=True)
        # Removing multimount double transcription records
        self.master_csv['AltCatalogNumber'] = self.master_csv['AltCatalogNumber'].apply(self.clean_herb_accession)

        self.master_csv = self.master_csv[self.master_csv['AltCatalogNumber'] != '']

        new_barcode_set = set(self.master_csv['Barcode'].unique())

        removed_set = barcode_set - new_barcode_set
        if len(removed_set) > 0:
            self.logger.warning(f"Some barcodes including [{removed_set}] dropped all records due to issues")
        # Re-indexing dataframe

    def clean_elevation(self):
        """clean_elevation: cleans elevation such that it contains only numeric data ordered from min to max.
                            Additionally adds in unit and removes unknowns.
                            Also checks against accidental transcriptions of collector number as altitude.
        """
        # Removing non-numeric punctuation
        min_elevation = remove_non_numerics(str(self.row['MinElevation'])).strip()
        max_elevation = remove_non_numerics(str(self.row['MaxElevation'])).strip()
        elevation_unit = str(self.row['OriginalElevationUnit']).strip()

        # Collector number to check accidental elevation transcription
        collector_number = remove_non_numerics(str(self.row['CollectorNumber'])).strip()

        # Moving incorrectly placed min elevation and emptying out
        if (pd.isna(min_elevation) or min_elevation.strip("0") == "") and pd.notna(max_elevation):
            min_elevation = max_elevation
            max_elevation = ''
        elif ((pd.isna(min_elevation) or min_elevation.strip("0")) == "" or min_elevation.startswith("0")) and \
                (pd.isna(max_elevation) or max_elevation.strip("0") == "" or max_elevation.startswith("0")):
            min_elevation = ''
            max_elevation = ''
            elevation_unit = ''
        elif (len(max_elevation) - len(min_elevation)) >= 2 and pd.notna(min_elevation):
            len_diff = len(max_elevation) - len(min_elevation)
            min_elevation = min_elevation + ('0' * len_diff)
        elif min_elevation == max_elevation:
            max_elevation = ''

        is_unknown = self.detect_is_empty(elevation_unit)

        if not str(min_elevation).endswith("0") and is_unknown:
            min_elevation = ''
            elevation_unit = ''
        elif str(min_elevation).endswith("0") and is_unknown:
            # Currently matched to highest possible altitude in North America in meters
            if int(min_elevation) > self.config.ELEV_UPPER_BOUND:
                elevation_unit = 'ft'
            else:
                elevation_unit = 'm'

        # emptying out single digit entries
        if ((len(min_elevation) <= 1 and not self.detect_is_empty(min_elevation))
                or len(min_elevation) > 5):
            min_elevation = ''
            max_elevation = ''
            elevation_unit = ''



        if elevation_unit and elevation_unit.lower() == "feet":
            elevation_unit = "ft"
        elif elevation_unit and elevation_unit.lower() == "meters":
            elevation_unit = "m"

        # Checking against collector number to prevent mistaken transcriptions.
        if str(min_elevation) == collector_number or str(max_elevation) == collector_number:
            min_elevation = ''
            max_elevation = ''
            elevation_unit = ''

        self.row['MinElevation'] = str(min_elevation) if elevation_unit is not None else np.nan
        self.row['MaxElevation'] = str(max_elevation) if elevation_unit is not None else np.nan
        self.row['OriginalElevationUnit'] = str(elevation_unit) if elevation_unit is not None else np.nan

    def clean_herb_accession(self, acc_num):
        """Removes accession numbers for mistakenly transcribed multi-mounts,
            and standardizes empty entries into [No Accession].
        """
        acc_num = str(acc_num).strip()
        if acc_num == "[No Accession]" or self.detect_is_empty(acc_num) or acc_num == "":
            acc_num = "[No Accession]"
        elif (len(remove_non_numerics(acc_num)) < len(acc_num)) or (len(remove_non_numerics(acc_num)) > 10):
            acc_num = ""
        return acc_num

    def regex_check_coord(self, coord: str, regex_pattern, max_num: int):
        if coord and not self.detect_is_empty(coord):
            # If regex pattern is found anywhere within the coord
            match = regex_pattern.search(coord)
            if match:
                township_number = int(match.group(1))
                if not (1 <= township_number <= max_num):
                    coord = ''
                else:
                    return coord
            else:
                coord = ''
        else:
            coord = ''
        return coord

    def clean_trs_utm_llm(self):
        """TRS/UTM cleaning across the DataFrame for only rows with trs/utm present"""
        system_prompt = self.read_in_prompt(filepath="prompts/trs_utm_prompt.txt")

        # Count max coord sets by column name pattern
        max_sets = sum(name.startswith("Township_") for name in self.master_csv.columns)
        if max_sets == 0:
            self.logger.info("No Township_* columns found; skipping clean_trs_utm_llm()")
            return

        cleaned_rows = []
        for _, row in self.master_csv.iterrows():
            row_dict = row.to_dict()
            row_dict = self._clean_trs_utm_row(row_dict, max_sets, system_prompt)
            cleaned_rows.append(row_dict)

        self.master_csv = pd.DataFrame(cleaned_rows)

    def _clean_trs_utm_row(self, row_dict: dict, max_sets: int, system_prompt: str) -> dict:
        """
        Clean TRS/UTM fields for a single row (dict).
        - Calls LLM only when coordinates_present_i indicates TRS/UTM and payload not all blank
        - Clears Quadrangle if TRS blank; clears Datum if UTM blank
        - Returns the updated row dict
        """
        for i in range(1, max_sets + 1):
            coord_presence_col = f"coordinates_present_{i}"
            coord_type = str(row_dict.get(coord_presence_col, "")).strip()

            is_trs = coord_type == "Yes - TRS (Township Range Section)"
            is_utm = coord_type == "Yes - UTM (Universal Transverse Mercator)"
            do_llm = is_trs or is_utm

            payload = {
                "Township": str(row_dict.get(f"Township_{i}", "")),
                "Range": str(row_dict.get(f"Range_{i}", "")),
                "Section": str(row_dict.get(f"Section_{i}", "")),
                "Quadrangle": str(row_dict.get(f"Quadrangle_{i}", "")),
                "Utm_zone": str(row_dict.get(f"Utm_zone_{i}", "")),
                "Utm_easting": str(row_dict.get(f"Utm_easting_{i}", "")),
                "Utm_northing": str(row_dict.get(f"Utm_northing_{i}", "")),
                "Datum": str(row_dict.get(f"Utm_datum_{i}", "")),
            }

            all_blank = all(self.detect_is_empty(v) for v in payload.values())

            # Run LLM if appropriate; otherwise keep as-is
            if do_llm and not all_blank:
                resp = self.send_to_llm(json.dumps(payload), system_prompt=system_prompt)
                if not isinstance(resp, dict):
                    self.logger.warning(f"coord set {i} - LLM failed: {resp}")
                    resp = payload.copy()

                self.logger.info(f"coord set {i} - LLM output: {resp}")
            else:
                resp = payload


            # Normalize and apply clearing rules
            township = resp.get("Township", "")
            range_ = resp.get("Range", "")
            section = resp.get("Section", "")
            quadrangle = resp.get("Quadrangle", "")
            utm_zone_r = resp.get("Utm_zone", "")
            utm_easting_r = resp.get("Utm_easting", "")
            utm_northing_r = resp.get("Utm_northing", "")
            datum_r = resp.get("Datum", "")

            trs_blank = all(self.detect_is_empty(x) for x in (township, range_, section))
            utm_blank = all(self.detect_is_empty(x) for x in (utm_zone_r, utm_easting_r, utm_northing_r))

            if trs_blank:
                quadrangle = ""
            if utm_blank:
                datum_r = ""

            # Write back
            row_dict[f"Township_{i}"] = township
            row_dict[f"Range_{i}"] = range_
            row_dict[f"Section_{i}"] = section
            row_dict[f"Quadrangle_{i}"] = quadrangle
            row_dict[f"Utm_zone_{i}"] = utm_zone_r
            row_dict[f"Utm_easting_{i}"] = utm_easting_r
            row_dict[f"Utm_northing_{i}"] = utm_northing_r
            row_dict[f"Utm_datum_{i}"] = datum_r

        return row_dict


    def clean_habitat_specimen_description_llm(self):
        """Cleans and separates unstructured strings into habitat and specimen description using a llm api"""
        cleaned_habitats = []
        cleaned_specimen_desc = []
        system_prompt = self.read_in_prompt(filepath=f"prompts/hab_spec_prompt.txt")

        for index, row in self.master_csv.iterrows():
            habitat = row['Remarks']
            specimen_description = row['Text1']

            if pd.isna(habitat) and pd.isna(specimen_description):
                cleaned_habitats.append("")
                cleaned_specimen_desc.append("")
                continue

            text = json.dumps({
                "habitat": habitat if isinstance(habitat, str) else "",
                "specimen_description": specimen_description if isinstance(specimen_description, str) else ""
            })
            structured_data = self.send_to_llm(text, system_prompt=system_prompt)
            self.logger.info(f"LLM returned: {structured_data}")

            # If response failed or isn't dict-like, return original
            if not isinstance(structured_data, dict):
                self.logger.warning(f"LLM did not return a valid dictionary: {structured_data}")
                cleaned_habitats.append("No dictionary returned")
                cleaned_specimen_desc.append("No dictionary returned")
            else:
                cleaned_habitats.append(structured_data.get("habitat", ""))
                cleaned_specimen_desc.append(structured_data.get("specimen_description", ""))

        self.master_csv['cleaned_habitat'] = cleaned_habitats
        self.master_csv['cleaned_spec_desc'] = cleaned_specimen_desc


    def send_to_llm(self, user_input, system_prompt):
        """building block function which posts the llm api request, assumes default llama3:70b"""
        url = f"{self.ollama_url}/api/chat"
        try:
            self.logger.info(f"Sending request to: {url}")
            response = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                data=json.dumps({
                    "model": "llama3:70b",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_input}
                    ],
                    "stream": False
                })
            )

            if response.status_code != 200:
                self.logger.error(f"Bad response: {response.status_code} - {response.text}")
                return "Error"

            response_json = response.json()
            raw_text = response_json.get("message", {}).get("content", "")

            parsed = self.clean_and_parse_json5(raw_text=raw_text)
            return parsed
        except Exception as e:
            self.logger.error(f"Error in querying Mistral: {e}")
            return "Error"

    def clean_and_parse_json5(self, raw_text):
        """parses json output of LLM functions"""
        try:
            # Remove markdown formatting (e.g., ```json)
            cleaned = raw_text.strip().lstrip("`").rstrip("`")
            # Find the first open brace
            start = cleaned.find("{")
            if start == -1:
                raise ValueError("No opening brace found")
            # Manually extract everything from the first brace
            json_candidate = cleaned[start:]
            # If there's no closing brace, add one
            if not json_candidate.strip().endswith("}"):
                json_candidate += "}"
            # Try parsing
            return json5.loads(json_candidate)
        except Exception as e:
            self.logger.error(f"Could not clean/parse JSON5 from text: {raw_text}")
            self.logger.error(f"Parsing error detail: {e}")
            return "Error"


    def has_matching_substring(self, row, column1, column2):
        fullname_parts = str(row[column1]).split()  # Split fullname into parts
        name_matched_parts = str(row[column2]).split()  # Split name_matched into parts
        # Check if any part in fullname matches name_matched (case-insensitive)
        return any(f.lower() == n.lower() for f in fullname_parts for n in name_matched_parts)

    def _safe_parse_coord(self, coord_string, coord_type, hemisphere="NorthWest"):
        """
        coord_type: 'lat' or 'lon'
        hemisphere: one of NorthWest, NorthEast, SouthWest, SouthEast (default = NorthWest)
        """
        try:
            val = parse_coordinate(str(coord_string), coord_type=coord_type)
            if val is None:
                return math.nan

            val = float(val)
            s = str(coord_string).strip().upper()

            HEMISPHERE_DEFAULTS = {
                "NorthWest": ("N", "W"),
                "NorthEast": ("N", "E"),
                "SouthWest": ("S", "W"),
                "SouthEast": ("S", "E"),
            }

            lat_default, lon_default = HEMISPHERE_DEFAULTS.get(
                hemisphere, ("N", "W")
            )

            if coord_type.lower() in ("lat", "latitude"):
                if "S" in s:
                    return -abs(val)
                elif "N" in s:
                    return abs(val)
                else:
                    return abs(val) if lat_default == "N" else -abs(val)

            elif coord_type.lower() in ("lon", "long", "longitude"):
                # Explicit E/W in the string takes precedence
                if "W" in s:
                    return -abs(val)
                elif "E" in s:
                    return abs(val)
                else:
                    return -abs(val) if lon_default == "W" else abs(val)
            return math.nan

        except Exception:
            return math.nan

    def flag_failed_numeric_conversion(self, row, matches):
        """flags rows where not all the verbatim lat longs converted succesfully to
           numeric lat longs for manual review
        """
        verb_count = 0
        num_count = 0

        if matches <= 0:
            return False

        for i in range(1, matches + 1):
            for col in (f"lat_verbatim_{i}", f"long_verbatim_{i}"):
                if col in row.index and not self.detect_is_empty(row[col]):
                    verb_count += 1

            for col in (f"lat_numeric_{i}", f"long_numeric_{i}"):
                if col in row.index and not self.detect_is_empty(row[col]):
                    num_count += 1

        if verb_count == 0:
            return False

        # Flag when counts don't match
        return verb_count != num_count



    def filter_lat_long_frame(self):
        """Only keeps lat/long columns from processing if they exist. Creates numeric_columns"""
        columns = [
            'lat_verbatim_1', 'long_verbatim_1',
            'lat_verbatim_2', 'long_verbatim_2',
            'lat_verbatim_3', 'long_verbatim_3',
        ]

        existing_columns = [col for col in columns if col in self.master_csv.columns]

        matches = len(existing_columns) // 2

        # build numeric counterparts for any discovered pair index i
        for i in range(1, matches + 1):
            lat_v = f'lat_verbatim_{i}'
            lon_v = f'long_verbatim_{i}'
            if lat_v in self.master_csv.columns:
                self.master_csv[f'lat_numeric_{i}'] = self.master_csv[lat_v].apply(
                    lambda x: self._safe_parse_coord(x, coord_type="latitude"))
                existing_columns.append(f'lat_numeric_{i}')
            if lon_v in self.master_csv.columns:
                self.master_csv[f'long_numeric_{i}'] = self.master_csv[lon_v].apply(
                    lambda x: self._safe_parse_coord(x, coord_type="longitude"))
                existing_columns.append(f'long_numeric_{i}')

        return self.master_csv[existing_columns].copy()

    def extract_coords_for_column_pair(self, coord_frame, lat_col, lon_col):
        """
        Converts latitude/longitude pair into a list of [lat, lon] lists,
        skipping any rows where lat or lon is NaN or unparseable.
        """
        coords = []
        for _, row in coord_frame.iterrows():
            try:
                lat = float(row[lat_col])
                lon = float(row[lon_col])
            except (ValueError, TypeError):
                # couldn't parse to float
                continue

            # drop NaNs
            if math.isnan(lat) or math.isnan(lon):
                continue

            coords.append([lat, lon])

        return coords

    def batch_query_gvs(
            self,
            coord_num: int,
            coord_frame: pd.DataFrame,
            api_url: str = "https://gvsapi.xyz/gvs_api.php",
            mode: str = "resolve",
            maxdist: float | None = 10,
            maxdistrel: float | None = 0.1
    ) -> pd.DataFrame | None:
        """
        Batch-queries the GVS API by building a payload of the form:
        {
          "opts": { "mode": "...", "maxdist": ..., "maxdistrel": ... },
          "data": [ [lat1, lon1], [lat2, lon2], … ]
        }
        """

        data = self.extract_coords_for_column_pair(
            coord_frame,
            f"lat_numeric_{coord_num}",
            f"long_numeric_{coord_num}"
        )
        if not data:
            logging.info(f"No valid coordinates for batch: {coord_num}")
            return None
        opts: dict[str, float | str] = {"mode": mode}
        if maxdist is not None:
            opts["maxdist"] = maxdist
        if maxdistrel is not None:
            opts["maxdistrel"] = maxdistrel

        payload = {
            "opts": opts,
            "data": data
        }

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "charset": "UTF-8"
        }

        try:
            resp = requests.post(api_url, headers=headers, data=json.dumps(payload))
            resp.raise_for_status()
            return pd.DataFrame(resp.json())

        except requests.exceptions.RequestException as e:
            logging.info(f"An error occurred: {e}")
            return None

    def clean_coords(self):
        """Loops through each lat_verbatim_X column and prints the GVS results."""
        coord_frame = self.filter_lat_long_frame()

        matches = sum(name.startswith("lat_verbatim_") for name in coord_frame.columns)

        for i in range(1, matches + 1):
            df = self.batch_query_gvs(coord_num=i, coord_frame=coord_frame)

            if df is None or df.empty:
                self.logger.warning(f"No valid results for lat_verbatim_{i}")
                continue

            df.reset_index(drop=True, inplace=True)

            coord_frame_subset = coord_frame[[f"lat_verbatim_{i}", f"long_verbatim_{i}"]].reset_index()

            df.rename(columns={'county': 'assigned_county'}, inplace=True)

            df = df[['assigned_county', 'latlong_err']]

            merged = pd.concat([coord_frame_subset, df], axis=1)

            for col in df.columns:
                cleaned_col = f"{col}_{i}" if col not in self.master_csv.columns else f"{col}_{i}_cleaned"
                self.master_csv.loc[merged["index"], cleaned_col] = merged[col].values

            self.logger.info(f"Merged cleaned coordinates for lat_verbatim_{i} into master_csv.")

        self.master_csv["failed_conversion"] = self.master_csv.apply(
            lambda row: self.flag_failed_numeric_conversion(row, matches),
            axis=1
        )

    def infer_column_types(self, df: pd.DataFrame, group_by: str = "subject_ids") -> list[str]:
        """
        Hardcoded column type assignments for run_on_dataframe().

        Each entry is "column_name:column_type"
        Types: 'same', 'select', 'text', 'noop', 'point'
        """
        # Hardcode mapping here
        column_assignment = {
            "Barcode": "same",
            "State": "same",
            "County": "same",
            "Country": "same",
            "CollectorNumber": "same",
            "classification_id": "text",
            "user_name": "noop",
            "user_id": "noop",
            "workflow_name": "noop",
            "workflow_version": "noop",
            "subject_ids": "same",
            "herbarium_code": "select",
            "AltCatalogNumber": "text",
            "workflow_id": "noop",
            "Remarks": "text",
            "Text1": "text",
            "MinElevation": "text",
            "MaxElevation": "text",
            "OriginalElevationUnit": "select"
        }

        coordinate_fields = {
            "coordinates_present": "select",
            "Township": "text",
            "Range": "text",
            "Section": "text",
            "Quadrangle": "text",
            "Utm_zone": "text",
            "Utm_easting": "text",
            "Utm_northing": "text",
            "utm_datum": "text",
            "lat_verbatim": "text",
            "long_verbatim": "text",
            "lat_numeric": "text",
            "long_numeric": "text",
            "lat_long_datum": "text"
        }

        for base_name, ctype in coordinate_fields.items():
            for i in range(1, 4):
                column_assignment[f"{base_name}_{i}"] = ctype


        # Produce list in "col:type" format for run_on_dataframe
        column_types = []
        for col in df.columns:
            if col == group_by:
                continue
            ctype = column_assignment.get(col, "text")  # default fallback
            column_types.append(f"{col}:{ctype}")

        return column_types

    def flag_only_one(self, col_list):

        """
         Flags rows True if, within each `group_col` group, *any* column in `col_list`
         has exactly one filled value (non-NaN and not just whitespace).
        """

        if not col_list:
            self.logger.warning("No columns supplied")
            self.master_csv["only_one_entry"] = False
            self.master_csv["only_one_entry_detail"] = ""
            return self.master_csv

        filled = {c: (self.master_csv[c].notna()) & (~self.master_csv[c].astype(str).str.strip().eq("")) for c in col_list}
        filled_df = pd.DataFrame(filled, index=self.master_csv.index)

        counts = filled_df.groupby(self.master_csv["Barcode"]).sum()
        singleton_cols_per_group = counts.apply(lambda s: [c for c, v in s.items() if v == 1], axis=1)

        self.master_csv["only_one_entry"] = self.master_csv["Barcode"].map(counts.eq(1).any(axis=1))

        self.master_csv["only_one_entry_detail"] = self.master_csv["Barcode"].map(lambda g: ",".join(
                                                   singleton_cols_per_group.get(g, [])))

    def reconcile_rows(self):
        """calls reconciler to perform final row combination"""
        column_types = self.infer_column_types(self.master_csv)
        unrec_df, rec_df = run_on_dataframe(
                            self.master_csv,
                            column_types=column_types,
                            format_choice="csv",
                            group_by="subject_ids" if "subject_ids" in self.master_csv.columns else self.master_csv.columns[0],
                            explanations=True,
                            summary_path=self.summary_path
                            )

        # remove suffix from reconciler
        rec_df.columns = rec_df.columns.str.replace(r'(_\d+)$', '', regex=True)

        return unrec_df, rec_df

    def run_all_methods(self):
        self.rename_columns()
        self.remove_records()
        for index, row in self.master_csv.iterrows():
            self.row = row
            self.index = index
            self.clean_elevation()
            self.master_csv.loc[self.row.name] = self.row

        self.clean_coords()
        self.clean_trs_utm_llm()
        self.clean_habitat_specimen_description_llm()


        os.makedirs(f"nfn_csv{os.path.sep}nfn_csv_output", exist_ok=True)

        output_base_name = os.path.splitext(os.path.basename(self.input_file))[0]

        self.summary_path = f"nfn_csv{os.path.sep}nfn_csv_output{os.path.sep}{output_base_name}_summary.html"

        self.master_csv.drop(columns=['classification_id', 'Remarks', 'Text1', ])

        self.master_csv.to_csv(
            f"nfn_csv{os.path.sep}nfn_csv_output{os.path.sep}{output_base_name}_unreconciled.csv",
            sep=',',
            quoting=csv.QUOTE_ALL,
            index=False
        )

        self.flag_only_one(col_list=["MinElevation", "Township_1", "Utm_zone_1", "lat_verbatim_1"])

        unrec_csv, rec_csv = self.reconcile_rows()

        rec_csv.to_csv(
            f"nfn_csv{os.path.sep}nfn_csv_output{os.path.sep}{output_base_name}_reconciled.csv",
            sep=',',
            quoting=csv.QUOTE_ALL,
            index=False
        )



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Runs checks on NFN csvs and returns a wrangled csv ready for upload.")

    parser.add_argument('-v', '--verbosity',
                        help='verbosity level. repeat flag for more detail',
                        default=0,
                        dest='verbose',
                        action='count')

    parser.add_argument("-i", "--input_file", nargs="?",
                        default=None,
                        help="name of nfn batch file")

    parser.add_argument("-l", "--log_level", nargs="?",
                        default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level (default: %(default)s)")

    parser.add_argument("-c", "--collection", nargs="?",
                        default="Botany_PIC", choices=["Botany_PIC", "iz", "ich"],
                        help="Collection type (default: %(default)s)")

    parser.add_argument('-hm', '--hemisphere', nargs="?",
                        default="NorthWest", choices=['NorthWest', 'NorthEast', 'SouthWest', 'SouthEast'],
                        help="the hemisphere quadrant the dataset is covering, to standardize +, - numeric lat/longs"
                        )


    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    picturae_csv_instance = NfnCsvCreate(coll=args.collection, input_file=args.input_file, logging_level=args.log_level,
                                         hemisphere=args.hemisphere)
    picturae_csv_instance.run_all_methods()
