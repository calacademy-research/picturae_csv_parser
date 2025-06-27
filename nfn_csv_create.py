import pandas as pd
import os
from get_configs import get_config
from sql_csv_utils import SqlCsvTools
from string_utils import remove_non_numerics
import numpy as np
import csv
import re
import logging
import json5
import json
import requests
import argparse
import math

class NfnCsvCreate():
    def __init__(self, coll, logging_level):

        self.logger = logging.getLogger("NfnCreatePicturae")
        self.logger.setLevel(logging_level)

        self.config = get_config(coll)

        self.row = None
        self.index = None

        self.master_csv = self.read_and_concat_csvs()

        self.datums = self.config.ACCEPTED_DATUMS

        self.sql_csv_tools = SqlCsvTools(config=self.config)

        self.ollama_url = self.config.OLLAMA_URL

        self.nominatum_dict = {}


    def read_and_concat_csvs(self):
        """reads in and concatenates all csvs in the nfn csv folder"""
        csv_folder = self.config.NFN_CSV_FOLDER
        if not os.path.exists(csv_folder):
            raise FileNotFoundError(f"The folder {csv_folder} does not exist.")

        csv_files = [os.path.join(csv_folder, f) for f in os.listdir(csv_folder)
                     if f.endswith(".csv") and os.path.isfile(os.path.join(csv_folder, f))]

        self.logger.info(f"Found {len(csv_files)} CSVs in {csv_folder}")

        master_csv = pd.concat(
            [pd.read_csv(csv, dtype=str, low_memory=False) for csv in csv_files],
            ignore_index=True
        )
        return master_csv


    def detect_unknown_values(self, string):
        if string.lower() in ['none', 'unknown', 'unkown', '', 'nan'] or pd.isna(string):
            return True
        else:
            return False

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

        is_unknown = self.detect_unknown_values(elevation_unit)

        if not str(min_elevation).endswith("0") and is_unknown:
            min_elevation = ''
            elevation_unit = ''
        elif str(min_elevation).endswith("0") and is_unknown:
            # Currently matched to highest possible altitude in North America in meters
            if int(min_elevation) > self.config.ELEV_UPPER_BOUND:
                elevation_unit = 'ft'
            else:
                elevation_unit = 'm'

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
        if acc_num == "[No Accession]" or self.detect_unknown_values(acc_num) or acc_num == "":
            acc_num = "[No Accession]"
        elif (len(remove_non_numerics(acc_num)) < len(acc_num)) or (len(remove_non_numerics(acc_num)) > 10):
            acc_num = ""
        return acc_num

    def regex_check_coord(self, coord: str, regex_pattern, max_num: int):
        if coord and not self.detect_unknown_values(coord):
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


    # Remember to improve this to capture last edge cases., possibly modify this to use an llm query

    def clean_TRS_LLM(self):
        cleaned_rows = []
        system_prompt = self.config.TRS_UTM_LLM_PROMPT

        for index, row in self.master_csv.iterrows():

            for i in [1, 2, 3]:  # extend if needed
                coord_presence_col = f"coordinates_present_{i}"
                coord_type = str(row.get(coord_presence_col, "")).strip()

                # Only process TRS or UTM
                if coord_type not in [
                    "Yes - TRS (Township Range Section)",
                    "Yes - UTM (Universal Transverse Mercator)"
                ]:
                    continue

                # Build input payload
                trs_payload = {
                    "Township": str(row.get(f"Township_{i}", "")),
                    "Range": str(row.get(f"Range_{i}", "")),
                    "Section": str(row.get(f"Section_{i}", "")),
                    "Quadrangle": str(row.get(f"Quadrangle_{i}", "")),
                    "Utm_zone": str(row.get(f"Utm_zone_{i}", "")),
                    "Utm_easting": str(row.get(f"Utm_easting_{i}", "")),
                    "Utm_northing": str(row.get(f"Utm_northing_{i}", "")),
                    "Datum": str(row.get(f"Utm_datum_{i}", ""))
                }

                # Check if there's any actual value to clean
                if all(v.strip() in ["", "nan", "NaN"] for v in trs_payload.values()):
                    continue

                # Send to LLM
                response = self.send_to_llm(json.dumps(trs_payload), system_prompt=system_prompt)
                self.logger.info(f"Row {index}, coord set {i} - LLM returned: {response}")

                if isinstance(response, dict):
                    key_mapping = {
                        "Township": f"Township_{i}",
                        "Range": f"Range_{i}",
                        "Section": f"Section_{i}",
                        "Quadrangle": f"Quadrangle_{i}",
                        "Utm_zone": f"Utm_zone_{i}",
                        "Utm_easting": f"Utm_easting_{i}",
                        "Utm_northing": f"Utm_northing_{i}",
                        "Datum": f"Utm_datum_{i}"
                    }
                    for llm_key, target_col in key_mapping.items():
                        if llm_key in response:
                            row[target_col] = response[llm_key]
                else:
                    self.logger.warning(f"Row {index}, coord set {i} - LLM failed: {response}")

            cleaned_rows.append(row)

        self.master_csv = pd.DataFrame(cleaned_rows)

    def clean_habitat_specimen_description_llm(self):
        cleaned_habitats = []
        cleaned_specimen_desc = []
        system_prompt = self.config.HAB_SPEC_LLM_PROMPT

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

    def filter_lat_long_frame(self):
        """Only keeps lat/long columns from processing if they exist."""
        columns = [
            'lat_verbatim_1', 'long_verbatim_1',
            'lat_verbatim_2', 'long_verbatim_2',
            'lat_verbatim_3', 'long_verbatim_3',
        ]
        existing_columns = [col for col in columns if col in self.master_csv.columns]
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
        Batch‑queries the GVS API by building a payload of the form:
        {
          "opts": { "mode": "...", "maxdist": ..., "maxdistrel": ... },
          "data": [ [lat1, lon1], [lat2, lon2], … ]
        }
        """

        # 1) pull out only the finite coords
        data = self.extract_coords_for_column_pair(
            coord_frame,
            f"lat_verbatim_{coord_num}",
            f"long_verbatim_{coord_num}"
        )
        if not data:
            print("No valid coordinates for batch", coord_num)
            return None

        # 2) build opts exactly as in R
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
            # 3) POST the JSON exactly as R does
            resp = requests.post(api_url, headers=headers, data=json.dumps(payload))
            resp.raise_for_status()
            return pd.DataFrame(resp.json())

        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
            return None

    def clean_coords(self):
        """Loops through each lat_verbatim_X column and prints the GVS results."""
        coord_frame = self.filter_lat_long_frame()
        matches = sum(name.startswith("lat_verbatim_") for name in coord_frame.columns)

        for i in range(1, matches + 1):
            df = self.batch_query_gvs(coord_num=i, coord_frame=coord_frame)
            print(f"Results for lat_verbatim_{i}:\n", df)

    def run_all_methods(self):
        self.rename_columns()
        self.remove_records()
        for index, row in self.master_csv.iterrows():
            self.row = row
            self.index = index
            self.clean_elevation()
            self.master_csv.loc[self.row.name] = self.row

        self.clean_coords()
        self.clean_TRS_LLM()
        self.clean_habitat_specimen_description_llm()

        os.makedirs(f"nfn_csv{os.path.sep}nfn_csv_output", exist_ok=True)

        self.master_csv.to_csv(
            f"nfn_csv{os.path.sep}nfn_csv_output{os.path.sep}final_nfn_combined.csv",
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

    parser.add_argument("-l", "--log_level", nargs="?",
                        default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level (default: %(default)s)")

    parser.add_argument("-c", "--collection", nargs="?",
                        default="Botany_PIC", choices=["Botany_PIC", "iz", "ich"],
                        help="Collection type (default: %(default)s)")

    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    picturae_csv_instance = NfnCsvCreate(coll=args.collection, logging_level=args.log_level)
    picturae_csv_instance.run_all_methods()
