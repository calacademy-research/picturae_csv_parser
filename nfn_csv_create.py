import pandas as pd
import os
from get_configs import get_config
from string_utils import remove_non_numerics
import numpy as np
import csv
import re
import logging
class NfnCsvCreate():
    def __init__(self, csv_name, coll):
        self.csv_name = csv_name

        self.row = None
        self.index = None

        self.logger = logging.getLogger(f'Client.' + self.__class__.__name__)

        self.master_csv = pd.read_csv(f"nfn_csv{os.path.sep}{csv_name}", dtype=str, low_memory=False)

        self.config = get_config(coll)

        self.datums = ["WGS", "NAD", "OSGRB", "ETRS", "ED", "GDA", "JGD", "Tokyo", "KGD",
                       "TWD", "BJS", "XAS", "GCJ", "BD", "PZ", "GTRF", "CGCS",
                       "CGS", "ITRF", "ITRS", "SAD"]


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
            'T18_Herbarium Code_1': 'modifier', # herbarium code
            'T17_Accession Number _1': 'AltCatalogNumber', # accession number
            'T20_Habitat_1': 'Remarks', # Habitat
            'T21_Plant Description_1': 'Text1', # specimen description
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

        col_order_list = list(filtered_col_dict.values())

        self.master_csv = self.master_csv.reindex(columns=col_order_list, fill_value=None)

    def remove_records(self):
        """remove_records: removes potential problem records before cleaning.
          Includes double transcriptions, banned user names."""
        #removing blacklist records
        blacklist = self.config.nfn_blacklist
        barcode_set = set(self.master_csv['Barcode'].unique())

        self.master_csv = self.master_csv[~self.master_csv['user_name'].isin(blacklist)].reset_index(drop=True)
        # removing multimount double transcription records
        self.master_csv['AltCatalogNumber'] = self.master_csv['AltCatalogNumber'].apply(self.clean_herb_accession)

        self.master_csv = self.master_csv[self.master_csv['AltCatalogNumber'] != '']

        new_barcode_set = set(self.master_csv['Barcode'].unique())

        removed_set = barcode_set - new_barcode_set
        if len(removed_set) > 0:
            self.logger.warning(f"Some barcodes including [{removed_set}] dropped all records due to issues")
        # re-indexing dataframe


    def clean_elevation(self):
        """clean_elevation: cleans elevation such that it contains only numeric data ordered from min to max.
                            additionally adds in unit and removes unknowns.
                            Also checks against accidental transcriptions of collector number as altitutde
        """
        #removing non numeric punctuation
        min_elevation = remove_non_numerics(str(self.row['MinElevation'])).strip()
        max_elevation = remove_non_numerics(str(self.row['MaxElevation'])).strip()
        elevation_unit = str(self.row['OriginalElevationUnit']).strip()

        # collector number to check accidental elevation transcription
        collector_number = remove_non_numerics(str(self.row['CollectorNumber'])).strip()

        # moving incorrectly placed min elevation and emptying out

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
            # currently matched to highest possible altitude in north america in meters (might think of finding altitude resolver)
            if int(min_elevation) > 6190:
                elevation_unit = 'ft'
            else:
                elevation_unit = 'm'

        if elevation_unit and elevation_unit.lower() == "feet":
            elevation_unit = "ft"
        elif elevation_unit and elevation_unit.lower() == "meters":
            elevation_unit = "m"

        # checking against collector number to prevent mistaken transcriptions.
        if str(min_elevation) == collector_number or str(max_elevation) == collector_number:
            min_elevation = ''
            max_elevation = ''
            elevation_unit = ''

        self.row['MinElevation'] = str(min_elevation) if elevation_unit is not None else np.nan
        self.row['MaxElevation'] = str(max_elevation) if elevation_unit is not None else np.nan
        self.row['OriginalElevationUnit'] = str(elevation_unit) if elevation_unit is not None else np.nan



    def clean_herb_accession(self, acc_num):
        """removes accessions numbers for mistakenly transcribed multi-mounts,
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
            # if regex pattern is anywhere within the coord
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


    def regex_check_TRS(self):
        """compares each TRS entry egainst regex """
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
            except:
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
        """compares each UTM entry against regex and maximum numeric range.
           Checks utm datum against list of acceptible datum codes.
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
            except:
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


    def gsv_coords(self):
        pass

    def match_two_records(self):
        pass

    def bias_towards_transcriber(self):
        pass




    def run_all_methods(self):
        self.rename_columns()
        self.remove_records()
        for index, row in self.master_csv.iterrows():
            self.row = row
            self.index = index
            self.clean_elevation()
            self.regex_check_TRS()
            self.master_csv.loc[self.row.name] = self.row

        self.master_csv.to_csv(f"nfn_csv{os.path.sep}nfn_csv_output{os.path.sep}"
                               f"final_nfn_{remove_non_numerics(self.csv_name)}.csv",
                               sep=',',  # Change to '\t' for tab, '|' for pipe, etc. if needed
                               quoting=csv.QUOTE_ALL,  # Quote all entries
                               index=False  # Set to True if you want to include the DataFrame index
                               )
        # self.clean_elevation()
        # self.clean_herb_accession()


#test

nfn_csv = NfnCsvCreate(csv_name="26925_From_Phlox_Gilia_unreconciled.csv", coll="Botany_PIC")

nfn_csv.run_all_methods()
