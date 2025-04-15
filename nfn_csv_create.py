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
from taxon_tools.BOT_TNRS import process_taxon_resolve

# Configure logging for the module. You can adjust level and formatting as needed.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)

class NfnCsvCreate():
    def __init__(self, csv_name, coll, config):
        self.csv_name = csv_name

        self.row = None
        self.index = None

        self.logger = logging.getLogger(f'Client.{self.__class__.__name__}')

        self.master_csv = pd.read_csv(f"nfn_csv{os.path.sep}{csv_name}", dtype=str, low_memory=False)

        self.config = get_config(coll)

        self.datums = ["WGS", "NAD", "OSGRB", "ETRS", "ED", "GDA", "JGD", "Tokyo", "KGD",
                       "TWD", "BJS", "XAS", "GCJ", "BD", "PZ", "GTRF", "CGCS",
                       "CGS", "ITRF", "ITRS", "SAD"]

        self.sql_csv_tools = SqlCsvTools(config=config)

        # self.ollama_url = "http://10.1.10.176:11434"
        self.ollama_url = "http://10.1.10.183:11434"

        self.nominatum_dict = {}

    def detect_unknown_values(self, string):
        if string.lower() in ['none', 'unknown', 'unkown', '', 'nan'] or pd.isna(string):
            return True
        else:
            return False

    def rename_columns(self):
        col_dict = {
            'Barcode': 'Barcode',
            'Country': 'Country',
            'State': 'State',
            'County': 'County',
            'Locality': 'Locality',
            'Genus': 'Genus',
            'Species': 'Species',
            'Epithet_1': 'Epithet_1',
            'subject_id': 'subject_id',
            'CollectorNumber': 'CollectorNumber',
            'classification_id': 'classification',
            'user_name': 'user_name',
            'T18_Herbarium Code_1': 'modifier',  # herbarium code
            'T17_Accession Number _1': 'AltCatalogNumber',  # accession number
            'T20_Habitat_1': 'Remarks',  # Habitat
            'T21_Plant Description_1': 'Text1',  # specimen description
            'T23_Elevation - Minimum_1': 'MinElevation',
            'T24_Elevation - Maximum_1': 'MaxElevation',
            'T25_Elevation Units_1': 'OriginalElevationUnit',
            'T5_Are there geographic coordinates present_1': "coordinates_present_1",
            'T12_Township_1': 'Township_1',
            'T13_Range_1': 'Range_1',
            'T14_Section_1': 'Section_1',
            'T27_Map Quadrangle_1': 'Quadrangle_1',
            'T7_UTM Zone_1': "Utm_zone_1",
            'T9_UTM Easting_1': "Utm_easting_1",
            'T8_UTM Northing_1': "Utm_northing_1",
            'T11_UTM - Datum_1': "Utm_datum_1",
            'T4_Latitude coordinates - Verbatim_1': 'lat_verbatim_1',
            'T3_Longitude coordinates - Verbatim _1': 'long_verbatim_1',
            'T6_Datum - Latitude, Longitude_1': 'lat_long_datum_1',
            'T16_Is there another set of coordinates that need to be transcribed?_1': "coordinates_present_2",
            'T12_Township_2': 'Township_2',
            'T13_Range_2': 'Range_2',
            'T14_Section_2': 'Section_2',
            'T27_Map Quadrangle_2': 'Quadrangle_2',
            'T7_UTM Zone_2': "Utm_zone_2",
            'T9_UTM Easting_2': "Utm_easting_2",
            'T8_UTM Northing_2': "Utm_northing_2",
            'T11_UTM - Datum_2': "Utm_datum_2",
            'T4_Latitude coordinates - Verbatim_2': 'lat_verbatim_2',
            'T3_Longitude coordinates - Verbatim _2': 'long_verbatim_2',
            'T6_Datum - Latitude, Longitude_2': 'lat_long_datum_2',
            'T16_Is there another set of coordinates that need to be transcribed?_2': "coordinates_present_3",
            'T12_Township_3': 'Township_3',
            'T13_Range_3': 'Range_3',
            'T14_Section_3': 'Section_3',
            'T27_Map Quadrangle_3': 'Quadrangle_3',
            'T7_UTM Zone_3': "Utm_zone_3",
            'T9_UTM Easting_3': "Utm_easting_3",
            'T8_UTM Northing_3': "Utm_northing_3",
            'T11_UTM - Datum_3': "Utm_datum_3",
            'T4_Latitude coordinates - Verbatim_3': 'lat_verbatim_3',
            'T3_Longitude coordinates - Verbatim _3': 'long_verbatim_3',
            'T6_Datum - Latitude, Longitude_3': 'lat_long_datum_3'
        }

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
        blacklist = self.config.nfn_blacklist
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
            if int(min_elevation) > 6190:
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

    # Remember to improve this to capture last edge cases.
    def regex_check_TRS(self):
        """Compares each TRS entry against regex."""
        township_regex = re.compile(r'^T?\s*(0?[1-9]|[1-9][0-9])(N|S)$', re.IGNORECASE)
        range_regex = re.compile(r'^R?\s*(0?[1-9]|[1-9][0-9])(E|W)$', re.IGNORECASE)
        section_regex = re.compile(r'^S?\s*(0?[1-9]|[1-2][0-9]|3[0-6])$', re.IGNORECASE)

        column_names = list(self.master_csv.columns)
        matches = sum([name.startswith("Township") for name in column_names])

        for i in range(1, matches + 1):
            try:
                trs_township = str(self.row.get(f"Township_{i}", "")).strip()
                trs_range = str(self.row.get(f"Range_{i}", "")).strip()
                trs_section = str(self.row.get(f"Section_{i}", "")).strip()
                trs_map_quadrangle = str(self.row.get(f"Quadrangle_{i}", "")).strip()
            except Exception as e:
                self.logger.error(f"Error accessing TRS fields: {e}")
                break

            if any(not pd.isna(x) and x is not None and not self.detect_unknown_values(str(x))
                   for x in [trs_township, trs_range, trs_section, trs_map_quadrangle]):
                trs_township = self.regex_check_coord(coord=trs_township, regex_pattern=township_regex, max_num=99)
                trs_range = self.regex_check_coord(coord=trs_range, regex_pattern=range_regex, max_num=99)
                trs_section = self.regex_check_coord(coord=trs_section, regex_pattern=section_regex, max_num=36)

                if not all(val in ["", "."] for val in [trs_township, trs_range, trs_section]):
                    if trs_map_quadrangle and not self.detect_unknown_values(trs_map_quadrangle):
                        pass
                    else:
                        trs_map_quadrangle = ""
                else:
                    trs_map_quadrangle = ""
            else:
                trs_township = ""
                trs_range = ""
                trs_section = ""
                trs_map_quadrangle = ""

            self.row[f'Township_{i}'] = trs_township
            self.row[f'Range_{i}'] = trs_range
            self.row[f'Section{i}'] = trs_section
            self.row[f'Quadrangle_{i}'] = trs_map_quadrangle

    def regex_check_UTM(self):
        """Compares each UTM entry against regex and maximum numeric range.
           Checks utm datum against list of acceptable datum codes.
        """
        zone_regex = re.compile(r'^\s*(0?[1-9]|[1-5][0-9]|60)\s*$')
        easting_regex = re.compile(r'^\s*([1-8][0-9]{5}|900000)$', re.IGNORECASE)
        northing_regex = re.compile(r'^\s*(\d{1,7}|[1-9]\d{6}|10000000)$', re.IGNORECASE)

        column_names = list(self.master_csv.columns)
        matches = sum([name.startswith("Township") for name in column_names])

        for i in range(1, matches + 1):
            try:
                zone = remove_non_numerics(str(self.row.get(f"Utm_zone_{i}", "")).strip())
                easting = remove_non_numerics(str(self.row.get(f"Utm_easting_{i}", "")).strip())
                northing = remove_non_numerics(str(self.row.get(f"Utm_northing_{i}", "")).strip())
                utm_datum = remove_non_numerics(str(self.row.get(f"Datum_{i}", "")).strip().upper())
            except Exception as e:
                self.logger.error(f"Error accessing UTM fields: {e}")
                break

            if any(not pd.isna(x) and x is not None and not self.detect_unknown_values(str(x))
                   for x in [zone, easting, northing]):
                zone = self.regex_check_coord(coord=zone, regex_pattern=zone_regex, max_num=60)
                easting = self.regex_check_coord(coord=easting, regex_pattern=easting_regex, max_num=900000)
                northing = self.regex_check_coord(coord=northing, regex_pattern=northing_regex, max_num=10000000)
                if not all(val in ["", "."] for val in [zone, easting, northing]):
                    if any(datum in utm_datum for datum in self.datums):
                        pass
                    else:
                        utm_datum = ""
            else:
                zone = ""
                easting = ""
                northing = ""
                utm_datum = ""

            self.row[f'Utm_zone_{i}'] = zone
            self.row[f'Utm_easting_{i}'] = easting
            self.row[f'Utm_section_{i}'] = northing
            self.row[f'Datum_{i}'] = utm_datum

    def clean_habitat_specimen_description_llm(self):
        cleaned_habitats = []
        cleaned_specimen_desc = []

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

            structured_data = self.send_to_llm(text)
            self.logger.info(f"LLM returned: {structured_data}")

            # If response failed or isn't dict-like, return original
            if not isinstance(structured_data, dict):
                self.logger.warning(f"LLM did not return a valid dictionary: {structured_data}")
                cleaned_habitats.append("No dictionary returned")
                cleaned_specimen_desc.append("No dictionary returned")
            else:
                cleaned_habitats.append(structured_data.get("habitat", ""))
                cleaned_specimen_desc.append(structured_data.get("specimen_description", ""))

        self.master_csv['cleaned_remarks'] = cleaned_habitats
        self.master_csv['cleaned_spec_desc'] = cleaned_specimen_desc

    def send_to_llm(self, user_input):
        url = f"{self.ollama_url}/api/chat"
        system_prompt = (
            "You are a label-cleaning AI that strictly extracts *verbatim* text for fields: `habitat`, `specimen_description` and 'locality'. "
            "You will be given a JSON object with two or three input fields: `habitat` and `specimen_description`, and sometimes 'locality' but not always locality. "
            "Each field may include mixed information. Your job is to remove all content that does not belong in each field according to the rules below.\n\n"
            "📌 DO NOT add or infer any new information. ONLY retain verbatim content. If a phrase is required for sentence completeness, you may retain it even if it's borderline.\n\n"
            "🔒 Follow these strict rules:\n\n"
            "▶️ VERBATIM ONLY:\n"
            "- Do not rephrase, summarize, or infer meaning.\n"
            "- Do not turn phrases into lists or categories.\n"
            "- Use exact phrases from the label text only.\n\n"
            "▶️ FIELD DEFINITIONS & RULES:\n"
            "**Habitat**:\n"
            "- Describes the physical environment where the specimen grows.\n"
            "- Include: substrate (e.g. 'dry sand', 'loamy soil'), associated species, vegetation type (e.g. 'open grassland'), floodplains, power lines, and life zones.\n"
            "- General terms like 'along road', 'hills', 'on canyon slopes', or 'trail edge' go here.\n"
            "- named Burns or fires, like 'Bean Camp burn' or 'area of Carr Fire' go here. Un-named burns can also be included\n"
            "- Include: Associated Species and phrases like 'growing alongside' or 'associated species'"
            "- ❌ Do NOT include place names, geographic features, road names, or phrases like 'near [named place]'. These belong to locality.\n"
            "- ❌ Do NOT include details about the plant itself like height, color, or flowers.\n\n"
            "**Specimen Description**:\n"
            "- Describes the physical features of the plant only, not taxonomy or scientific name or author.\n"
            "- Include: size, color, shape, condition, maturity, flowers, inflorescence, abundance (e.g. 'common', 'many in bloom', 'rare', 'locally common/rare' etc.), and chromosome count like 'n=14'.\n"
            "- ❌ Do NOT include habitats or locality descriptions (e.g. 'grassland', 'near Glen Alpine').\n"
            "- ❌ Do NOT include place names.\n\n"
            "- ❌ Do NOT include Scientific Name or Author e.g. Eriastrum sapphirinum (Eastw.) or collomia linearis etc ...\n"
            "**Locality**:\n"
            "- Includes specific named places: cities, roads, parks, regions, mountain ranges, distances or bearings (e.g. '3 miles west of Jacumba').\n"
            "- ❌ Do NOT include general environments like 'grassland', 'floodplain', or 'roadside'.\n"
            "- ❌ Do NOT include plant traits or descriptions.\n\n"
            "▶️ FORMATTING:\n"
            "- Return a **valid JSON object**, like: {\"habitat\": \"...\", \"specimen_description\": \"...\", \"locality\": \"...\"}\n"
            "- If a field is missing, return an empty string: \"field\": \"\"\n"
            "- Do not wrap in markdown or include extra explanation.\n\n"
            "▶️ EXAMPLES:\n"
            "- Input: 'Dry sandy soil under sagebrush. Near McGee Creek. Small annual herb with yellow flowers.'\n"
            "- Output: {\"habitat\": \"Dry sandy soil under sagebrush.\", \"specimen_description\": \"Small annual herb with yellow flowers.\", \"locality\": \"Near McGee Creek.\"}\n"
            "- Input: 'Along roadside near Mono Co. border. Red flowers. Open grassland with scattered shrubs.'\n"
            "- Output: {\"habitat\": \"Along roadside. Open grassland with scattered shrubs.\", \"specimen_description\": \"Red flowers.\", \"locality\": \"Near Mono Co. border.\"}\n"
        )
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

    def clean_lat_long(self):
        pass

    def gsv_coords(self):
        pass

    def match_two_records(self):
        pass

    def bias_towards_transcriber(self):
        pass

    def has_matching_substring(self, row, column1, column2):
        fullname_parts = str(row[column1]).split()  # Split fullname into parts
        name_matched_parts = str(row[column2]).split()  # Split name_matched into parts
        # Check if any part in fullname matches name_matched (case-insensitive)
        return any(f.lower() == n.lower() for f in fullname_parts for n in name_matched_parts)

    def run_all_methods(self):
        self.rename_columns()
        self.remove_records()
        for index, row in self.master_csv.iterrows():
            self.row = row
            self.index = index
            self.clean_elevation()
            self.regex_check_TRS()
            self.regex_check_UTM()
            self.clean_lat_long()
            self.master_csv.loc[self.row.name] = self.row

        self.clean_habitat_specimen_description_llm()
        self.master_csv.to_csv(
            f"nfn_csv{os.path.sep}nfn_csv_output{os.path.sep}final_nfn_{remove_non_numerics(self.csv_name)}.csv",
            sep=',',       # Change to '\t' for tab, '|' for pipe, etc. if needed
            quoting=csv.QUOTE_ALL,  # Quote all entries
            index=False    # Set to True if you want to include the DataFrame index
        )

# Test code (uncomment to run)
pic_config = get_config("Botany_PIC")
nfn_csv = NfnCsvCreate(csv_name="26925_From_Phlox_Gilia_unreconciled.csv", coll="Botany_PIC", config=pic_config)
nfn_csv.run_all_methods()
