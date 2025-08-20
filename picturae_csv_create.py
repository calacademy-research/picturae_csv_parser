"""picturae_csv_create: this file is for wrangling and creating the dataframe
   and csv with the parsed fields required for upload, in picturae_import.
   Uses TNRS (Taxonomic Name Resolution Service) in taxon_check/test_TNRS.R
   to catch spelling mistakes, mis-transcribed taxa.
   Source for taxon names at IPNI (International Plant Names Index): https://www.ipni.org/ """
import argparse
import csv
import os.path
from taxon_parse_utils import *
from gen_import_utils import *
from string_utils import *
from os import path
from sql_csv_utils import SqlCsvTools
from specify_db import SpecifyDb
import logging
from get_configs import get_config
from taxon_tools.BOT_TNRS import iterate_taxon_resolve
from image_client import ImageClient
starting_time_stamp = datetime.now()

pd.set_option('future.no_silent_downcasting', True)

class IncorrectTaxonError(Exception):
    pass

class InvalidFilenameError(Exception):
    pass


class CsvCreatePicturae:
    def __init__(self, config, tnrs_ignore, covered_ignore, logging_level):
        self.tnrs_ignore = str_to_bool(tnrs_ignore)
        self.covered_ignore = str_to_bool(covered_ignore)
        self.picturae_config = config
        self.specify_db_connection = SpecifyDb(self.picturae_config)
        self.image_client = ImageClient(config=self.picturae_config)
        self.logger = logging.getLogger("CsvCreatePicturae")
        self.logger.setLevel(logging_level)

        self.init_all_vars()

        self.run_all()

    def get_first_digits_from_filepath(self, filepath, field_size=9):
        basename = os.path.basename(filepath)
        ints = re.findall(r'\d+', basename)
        if len(ints) == 0:
            raise InvalidFilenameError("Can't get barcode from filename")
        int_digits = int(ints[0])
        string_digits = f"{int_digits}"
        string_digits = string_digits.zfill(field_size)
        self.logger.debug(f"extracting digits from {filepath} to get {string_digits}")
        return string_digits


    def init_all_vars(self):
        """init_all_vars:to use for testing and decluttering init function,
                            initializes all class level variables  """

        self.cover_list = []

        self.sheet_list = []

        self.path_prefix = self.picturae_config.PREFIX

        self.dir_path = self.picturae_config.DATA_FOLDER + "csv_batch"



        # setting up alternate csv tools connections

        self.sql_csv_tools = SqlCsvTools(config=self.picturae_config, logging_level=self.logger.getEffectiveLevel())


        # intializing parameters for database upload
        init_list = ['taxon_id', 'barcode',
                     'collector_number', 'collecting_event_guid',
                     'collecting_event_id',
                     'determination_guid', 'collection_ob_id', 'collection_ob_guid',
                     'name_id', 'family', 'gen_spec_id', 'family_id',
                     'records_dropped']

        for param in init_list:
            setattr(self, param, None)


    def file_present(self):
        """file_present:
           checks if correct filepaths in working directory,
           checks if file is on input date
           checks if file folder is present.
           uses self.use_date to decide which folders to check
           args:
                none
        """

        to_current_directory()

        dir_sub = os.path.isdir(self.dir_path)

        if dir_sub is True:

                sheet_count = 0
                cover_count = 0

                for root, dirs, files in os.walk(self.dir_path):
                    for file in files:
                        file_string = file.lower()
                        if "sheet_cp1" in file_string:
                            sheet_count += 1
                            self.sheet_list.append(file)
                        elif "cover_cp1" in file_string:
                            cover_count += 1
                            self.cover_list.append(file)
                        else:
                            self.logger.info(f"csv {file} file does not fit format , skipping")
                if sheet_count != cover_count:
                    raise ValueError(f"Count of Sheet CSVs and Cover CSVs do not match {sheet_count} != {cover_count}")
                else:
                    self.logger.info("Sheet and Cover CSVs exist!")
        else:
            raise ValueError(f"picturae csv subdirectory not present")


    def csv_read_path(self, csv_level: str):
        """Reads in CSV data for given level and date.
        Args:
            csv_level (str): "COVER" or "SHEET" indicating the level of data.
        """
        dataframes = []

        if csv_level == "COVER":
            data_list = self.cover_list
        elif csv_level == "SHEET":
            data_list = self.sheet_list
        else:
            raise ValueError("Invalid csv_level value. It must be 'COVER' or 'SHEET'.")

        for csv_path in data_list:
            csv_path = self.dir_path + f"{os.path.sep}" + csv_path
            df = read_csv_file(csv_path)
            if " " in df.columns[0]:
                df = standardize_headers(df)
            dataframes.append(df)

        combined_csv = pd.concat(dataframes, ignore_index=True)

        if len(combined_csv) > 0:
            return combined_csv
        else:
            raise ValueError("The resulting DataFrame is empty; no data was loaded.")


    def csv_merge_and_clean(self):
        """csv_merge_and_clean:
                reads, merges and data wrangles the set of folder and specimen csvs
        """
        fold_csv, spec_csv = self.read_folder_and_specimen_csvs()

        spec_csv = self.fill_duplicate_barcodes(spec_csv=spec_csv)

        self.merge_folder_and_specimen_csvs(fold_csv, spec_csv)

        self.remove_duplicate_barcodes()


    def drop_common_columns(self, csv: pd.DataFrame, folder=False):
        """drops columns duplicate between sheet and cover csvs"""

        drop_list = ['APPLICATION-ID', 'OBJECT-TYPE', 'APPLICATION-BATCH', 'FEEDBACK-ALEMBO', 'FEEDBACK-CALIFORNIA']
        if folder is True:
            drop_list = drop_list + ['SPECIMEN-BARCODE', 'PATH-JPG', 'CSV-BATCH']

        csv.drop(columns=drop_list, inplace=True)

        return csv

    def read_folder_and_specimen_csvs(self):
        """read the folder and specimen CSVs into the environment.
            args:
                none
        """
        fold_csv = self.csv_read_path(csv_level="COVER")
        spec_csv = self.csv_read_path(csv_level="SHEET")

        # Set the folder barcode column to match the specimen barcode
        fold_csv['FOLDER-BARCODE'] = fold_csv['SPECIMEN-BARCODE']

        self.drop_common_columns(fold_csv, folder=True)
        self.drop_common_columns(spec_csv)

        difference = set(fold_csv['FOLDER-BARCODE']) - set(spec_csv['FOLDER-BARCODE'])

        if len(difference) > 0:
            self.logger.warning(f"Following folder barcodes not in specimen csv {difference}")

        return fold_csv, spec_csv


    def fill_duplicate_barcodes(self, spec_csv):
        """Populate barcode and Parent barcode based on duplicate records in notes section.
           args:
                spec_csv: the specimen level dataframe
        """
        spec_csv['SPECIMEN-BARCODE'] = spec_csv['SPECIMEN-BARCODE'].apply(remove_barcode_suffix)

        is_duplicate = spec_csv['NOTES'].astype(str).str.contains(r'\d', regex=True)
        spec_csv['DUPLICATE'] = is_duplicate
        spec_csv['PARENT-BARCODE'] = ''

        spec_csv.loc[is_duplicate, 'PARENT-BARCODE'] = spec_csv.loc[is_duplicate, 'SPECIMEN-BARCODE']
        spec_csv.loc[is_duplicate, 'SPECIMEN-BARCODE'] = spec_csv.loc[
            is_duplicate, 'NOTES'].apply(remove_non_numerics)

        spec_csv = fill_missing_folder_barcodes(df=spec_csv, spec_bar="SPECIMEN-BARCODE",
                                                fold_bar='FOLDER-BARCODE', parent_bar="PARENT-BARCODE")

        spec_csv = self.update_duplicate_notes(spec_csv=spec_csv)

        return spec_csv

    def update_duplicate_notes(self, spec_csv):
        """Creates grouped list of barcodes that share the same parent barcode and applies to the notes section"""
        # Group by parent barcode
        grouped = spec_csv.groupby('PARENT-BARCODE')['SPECIMEN-BARCODE'].apply(list).reset_index()
        grouped.columns = ['PARENT-BARCODE', 'SPECIMEN-BARCODES']

        # Create a dictionary mapping parent barcodes to specimen barcodes
        barcode_dict = dict(zip(grouped['PARENT-BARCODE'], grouped['SPECIMEN-BARCODES']))

        # Apply the parse function to update the 'NOTES' column
        spec_csv = self.parse_duplicate_notes(spec_csv=spec_csv, barcode_dict=barcode_dict)

        return spec_csv

    def parse_duplicate_notes(self, spec_csv, barcode_dict):
        """Parses a new aggregate duplicate note for barcodes that share the same parent barcode."""
        self.logger.info(f"{barcode_dict}")
        for parent_barcode, specimen_barcodes in barcode_dict.items():
            if parent_barcode:
                common_list = [parent_barcode] + barcode_dict[parent_barcode]
                total_barcodes = len(common_list)

                # Update NOTES for each specimen barcode
                for barcode in common_list:

                    other_barcodes = [b for b in common_list if b != barcode]

                    joined_barcodes = f"[{', '.join(other_barcodes)}]"

                    note_message = f"Multi-mount of {total_barcodes} barcodes. See also {joined_barcodes}."

                    spec_csv.loc[spec_csv['SPECIMEN-BARCODE'] == barcode, 'NOTES'] = note_message

            else:
                pass
        return spec_csv


    def merge_folder_and_specimen_csvs(self, fold_csv, spec_csv):
        """define self.record_full, Merge the folder and specimen CSVs and fill missing values.
            args:
                fold_csv: the folder level csv
                spec_csv: the specimen level csv
        """

        self.record_full = pd.merge(fold_csv, spec_csv, on="FOLDER-BARCODE")
        self.record_full.fillna(np.nan, inplace=True)
        self.record_full.rename(columns={"NOTES_x": "cover_notes", "NOTES_y": "sheet_notes"}, inplace=True)

        spec_difference = set(spec_csv['SPECIMEN-BARCODE']) - set(self.record_full['SPECIMEN-BARCODE'])

        # checking for specimen barcodes not matched to folder barcode
        if spec_difference:
            spec_difference = list(spec_difference)
            spec_difference.sort(key=lambda x: int(x) if x.isdigit() else float('inf'))

            filtered_spec_csv = spec_csv[spec_csv['SPECIMEN-BARCODE'].isin(spec_difference)]
            csv_batch_unmatch = filtered_spec_csv['CSV-BATCH'].unique()

            raise ValueError(f"In the following batches {sorted(set(csv_batch_unmatch))}, the"
                             f" following barcodes not matched to a folder {sorted(set(spec_difference))}")


    def remove_duplicate_barcodes(self):
        """Removing and saving rows with improperly marked duplicate records for further visual QC"""

        merge_len = len(self.record_full)

        # checking for improperly marked duplicates where specimen barcode is doubled instead of replaced
        duplicates = self.record_full[self.record_full.duplicated(subset='SPECIMEN-BARCODE', keep=False)]

        # where specimen barcode is duplicated, but collector-number is NOT duplicated.
        unmarked_dupes = duplicates[
            duplicates.duplicated(subset=['SPECIMEN-BARCODE', 'COLLECTOR-NUMBER'], keep=False) == False]

        unmarked_all = self.record_full[
            self.record_full['SPECIMEN-BARCODE'].isin(unmarked_dupes['SPECIMEN-BARCODE'])]

        # checking for duplicate rows
        self.record_full = self.record_full.drop(unmarked_dupes.index)
        self.record_full = self.record_full.drop_duplicates()

        # getting range of csv dates and writing unmarked duplicates to csv
        batch_date_list = self.record_full['CSV-BATCH'].apply(extract_digits, args=(8,))

        # re-assigning date_use to a range of dates
        self.date_range = f"{batch_date_list.min()}_{batch_date_list.max()}"

        if len(unmarked_all) > 0:
            unmarked_all.to_csv(f'picturae_csv/csv_batch/PIC_upload/spec_dup_{self.date_range}.csv',
                                quoting=csv.QUOTE_NONNUMERIC, index=False)

        unique_len = len(self.record_full)

        if merge_len > unique_len:
            self.logger.error(f"Detected {merge_len - unique_len} duplicate records")


    def csv_colnames(self):
        """csv_colnames: function to be used to rename columns to DB standards.
           args:
                none"""

        col_dict = {
                     'CSV-BATCH': 'CSV_batch',
                     'FOLDER-BARCODE': 'folder_barcode',
                     'SPECIMEN-BARCODE': 'CatalogNumber',
                     'PARENT-BARCODE': 'parent_CatalogNumber',
                     'PATH-JPG': 'image_path',
                     'LABEL-IS-MOSTLY-HANDWRITTEN': 'mostly_handwritten',
                     'TAXON-ID': 'taxon_id',
                     'FAMILY': 'Family',
                     'GENUS': 'Genus',
                     'SPECIES': 'Species',
                     'QUALIFIER': 'qualifier',
                     'RANK-1': 'Rank 1',
                     'EPITHET-1': 'Epithet 1',
                     'RANK-2': 'Rank 2',
                     'EPITHET-2': 'Epithet 2',
                     'cover_notes': 'cover_notes',
                     'HYBRID': 'Hybrid',
                     'AUTHOR': 'Author',
                     'COLLECTOR-NUMBER': 'collector_number',
                     'COLLECTOR-ID-1': 'agent_id1',
                     'COLLECTOR-FIRST-NAME-1': 'collector_first_name1',
                     'COLLECTOR-MIDDLE-NAME-1': 'collector_middle_name1',
                     'COLLECTOR-LAST-NAME-1': 'collector_last_name1',
                     'COLLECTOR-ID-2': 'agent_id2',
                     'COLLECTOR-FIRST-NAME-2': 'collector_first_name2',
                     'COLLECTOR-MIDDLE-NAME-2': 'collector_middle_name2',
                     'COLLECTOR-LAST-NAME-2': 'collector_last_name2',
                     'COLLECTOR-ID-3': 'agent_id3',
                     'COLLECTOR-FIRST-NAME-3': 'collector_first_name3',
                     'COLLECTOR-MIDDLE-NAME-3': 'collector_middle_name3',
                     'COLLECTOR-LAST-NAME-3': 'collector_last_name3',
                     'COLLECTOR-ID-4': 'agent_id4',
                     'COLLECTOR-FIRST-NAME-4': 'collector_first_name4',
                     'COLLECTOR-MIDDLE-NAME-4': 'collector_middle_name4',
                     'COLLECTOR-LAST-NAME-4': 'collector_last_name4',
                     'COLLECTOR-ID-5': 'agent_id5',
                     'COLLECTOR-FIRST-NAME-5': 'collector_first_name5',
                     'COLLECTOR-MIDDLE-NAME-5': 'collector_middle_name5',
                     'COLLECTOR-LAST-NAME-5': 'collector_last_name5',
                     'LOCALITY-ID': 'locality_id',
                     'COUNTRY': 'Country',
                     'STATE-LOCALITY': 'State',
                     'COUNTY': 'County',
                     'PRECISE-LOCALITY': 'locality',
                     'VERBATIM-DATE': 'verbatim_date',
                     'START-DATE-MONTH': 'start_date_month',
                     'START-DATE-DAY': 'start_date_day',
                     'START-DATE-YEAR': 'start_date_year',
                     'END-DATE-MONTH': 'end_date_month',
                     'END-DATE-DAY': 'end_date_day',
                     'END-DATE-YEAR': 'end_date_year',
                     'sheet_notes': 'sheet_notes',
                     'DUPLICATE': 'duplicate'
                    }


        col_order_list = []
        for key, value in col_dict.items():
            col_order_list.append(key)

        self.record_full = self.record_full.reindex(columns=col_order_list)

        # comment out before committing, code to create simple manifests
        # self.record_full['PATH-JPG'] = self.record_full['PATH-JPG'].apply(os.path.basename)
        #
        self.record_full.rename(columns=col_dict, inplace=True)

        # self.record_full.to_csv(f'picturae_csv/csv_batch/PIC_upload/master_db.csv',
        #                         quoting=csv.QUOTE_NONNUMERIC, index=False)
        #
        # self.logger.info("merged csv written")

    def missing_data_masks(self):
        """missing_data_masks: create masks and filtered csvs for each kind of relevant missing data to flag.
            returns:
                four filtered csvs --> missing_rank_csv, missing_geography_csv, missing_label_csv, invalid_date_csv
        """
        # flags in missing rank columns when > 1 infra-specific rank.
        rank1_missing = (self.record_full['Rank 1'].isna() | (self.record_full['Rank 1'] == '')) & \
                        (self.record_full['Epithet 1'].notna() & (self.record_full['Epithet 1'] != ''))

        rank2_missing = (self.record_full['Rank 2'].isna() | (self.record_full['Rank 2'] == '')) & \
                        (self.record_full['Epithet 2'].notna() & (self.record_full['Epithet 2'] != ''))

        missing_rank_csv = self.record_full.loc[rank1_missing & rank2_missing]

        # flags missing family in column
        missing_family = (self.record_full['Family'].isna() | (self.record_full['Family'] == '') |
                          (self.record_full['Family'].isnull()))

        missing_family_csv = self.record_full.loc[missing_family]

        # flags if missing higher geography
        missing_geography = (self.record_full['Country'].isna() | (self.record_full['Country'] == '') |
                             (self.record_full['Country'].isnull()))

        missing_geography_csv = self.record_full.loc[missing_geography]

        # flags if label is covered or folded.
        missing_label = ["covered" in str(row).lower() or "folded" in str(row).lower()
                         for row in self.record_full['sheet_notes']]

        missing_label_csv = self.record_full.loc[missing_label]

        # flags incorrect start date and end date
        invalid_start_date = ~self.record_full['start_date'].apply(validate_date)
        invalid_end_date = ~self.record_full['end_date'].apply(validate_date)

        invalid_date_mask = invalid_start_date | invalid_end_date
        invalid_date_csv = self.record_full.loc[invalid_date_mask]

        # flags verbatim date too long greater than 50 char and stores them in new label_data column

        invalid_verbatim_mask = self.record_full["verbatim_date"].str.len() > 50

        # adding lable data and new genus boolean
        self.record_full['label_data'] = ""
        self.record_full['new_genus'] = False

        self.record_full.loc[invalid_verbatim_mask, 'label_data'] = self.record_full.loc[
            invalid_verbatim_mask, 'verbatim_date']

        invalid_verbatim_csv = self.record_full.loc[invalid_verbatim_mask]

        return (missing_rank_csv, missing_family_csv, missing_geography_csv, missing_label_csv, invalid_date_csv, \
               invalid_verbatim_csv)

    def flag_missing_data(self):

        missing_rank_csv, missing_family_csv, missing_geography_csv, \
            missing_label_csv, invalid_date_csv, invalid_verbatim_csv = self.missing_data_masks()

        data_flag_dict = {"missing_rank": missing_rank_csv, "missing_family": missing_family_csv,
                          "missing_geography": missing_geography_csv, "missing_label": missing_label_csv,
                          "invalid_date": invalid_date_csv, "invalid_verbatim": invalid_verbatim_csv}

        message_dict = {
            "missing_rank": "Taxonomic names with 2 missing ranks at covers:",
            "missing_family": "Rows missing taxonomic family at barcodes:",
            "missing_geography": "Rows missing higher geography at barcodes:",
            "missing_label": "Label covered or folded at barcodes:",
            "invalid_date": "Invalid dates at:",
            "invalid_verbatim": "Verbatim date too long at:",
        }
        # flag missing and incorrect data
        message = ""
        for key, csv_data in data_flag_dict.items():
            if key == "missing_label" and self.covered_ignore:
                continue
            if len(csv_data) > 0:
                csv_data = csv_data.sort_values(by=['CSV_batch', 'CatalogNumber'])
                if key in ["missing_rank", "missing_family"]:
                    item_set = sorted(set(csv_data['folder_barcode']))
                    batch_set = sorted(set(csv_data['CSV_batch']))
                else:
                    item_set = sorted(set(csv_data['CatalogNumber']))
                    batch_set = sorted(set(csv_data['CSV_batch']))

                message += message_dict[key]
                message += f" {item_set} in batches {batch_set}\n\n"
        if message:
            raise ValueError(message.strip())

    def taxon_concat(self, row):
        """taxon_concat:
                parses taxon columns to check taxon database, adds the Genus species, ranks, and Epithets,
                in the correct order, to create new taxon fullname in self.fullname. so that can be used for
                database checks.
            args:
                row: a row from a csv file containing taxon information with correct column names

        """
        hyb_index = self.record_full.columns.get_loc('Hybrid')
        is_hybrid = row.iloc[hyb_index]

        # defining empty strings for parsed taxon substrings
        full_name = ""
        tax_name = ""
        first_intra = ""
        gen_spec = ""
        hybrid_base = ""

        gen_index = self.record_full.columns.get_loc('Genus')
        genus = row.iloc[gen_index]

        column_sets = [
            ['Genus', 'Species', 'Rank 1', 'Epithet 1', 'Rank 2', 'Epithet 2'],
            ['Genus', 'Species', 'Rank 1', 'Epithet 1'],
            ['Genus', 'Species']
        ]

        for columns in column_sets:
            for column in columns:
                index = self.record_full.columns.get_loc(column)
                if pd.notna(row.iloc[index]) and row.iloc[index] != '':
                    if columns == column_sets[0]:
                        full_name += f" {row.iloc[index]}"
                    elif columns == column_sets[1]:
                        first_intra += f" {row.iloc[index]}"
                    elif columns == column_sets[2]:
                        gen_spec += f" {row.iloc[index]}"

        full_name = full_name.strip()
        first_intra = first_intra.strip()
        gen_spec = gen_spec.strip()
        # creating taxon name
        # creating temporary string in order to parse taxon names without qualifiers
        separate_string = remove_qualifiers(full_name)

        taxon_strings = separate_string.split()

        second_epithet_in = row.iloc[self.record_full.columns.get_loc('Epithet 2')]
        first_epithet_in = row.iloc[self.record_full.columns.get_loc('Epithet 1')]
        spec_in = row.iloc[self.record_full.columns.get_loc('Species')]
        genus_in = row.iloc[self.record_full.columns.get_loc('Genus')]
        # changing name variable based on condition

        if pd.notna(second_epithet_in) and second_epithet_in != '':
            tax_name = remove_qualifiers(second_epithet_in)
        elif pd.notna(first_epithet_in) and first_epithet_in != '':
            tax_name = remove_qualifiers(first_epithet_in)
        elif pd.notna(spec_in) and spec_in != '':
            tax_name = remove_qualifiers(spec_in)
        elif pd.notna(genus_in) and genus_in != '':
            tax_name = remove_qualifiers(genus_in)
        else:
            return ValueError('missing taxon in row')

        if is_hybrid is True:
            if first_intra == full_name:
                if "var." in full_name or "subsp." in full_name or " f." in full_name or "subf." in full_name:
                    hybrid_base = full_name
                    full_name = " ".join(taxon_strings[:2])
                elif full_name != genus and full_name == gen_spec:
                    hybrid_base = full_name
                    full_name = taxon_strings[0]
                elif full_name == genus:
                    hybrid_base = full_name
                    full_name = full_name
                else:
                    self.logger.error('hybrid base not found')

            elif len(first_intra) != len(full_name):
                if "var." in full_name or "subsp." in full_name or " f." in full_name or "subf." in full_name:
                    hybrid_base = full_name
                    full_name = " ".join(taxon_strings[:4])
                else:
                    pass

        return str(gen_spec), str(full_name), str(first_intra), str(tax_name), str(hybrid_base)


    def col_clean(self):
        """parses and cleans dataframe columns until ready for upload.
            runs dependent function taxon concat
        """
        # concatenate date

        for col_name in list(["start", "end"]):
            self.record_full[f'{col_name}_date'] = self.record_full.apply(
                lambda row: format_date_columns(row[f'{col_name}_date_year'],
                                                row[f'{col_name}_date_month'], row[f'{col_name}_date_day']), axis=1)

        for colname in ['verbatim_date', 'locality', 'collector_number']:
            self.record_full[colname] = self.record_full[colname].apply(
                lambda x: replace_apostrophes(x).strip() if isinstance(x, str) else x
            )

        # flagging missing data
        self.flag_missing_data()


        # converting hybrid column to true boolean

        self.record_full['Hybrid'] = self.record_full['Hybrid'].apply(str_to_bool)

        # concatenating year, month, day columns into start/end date columns

        # Replace '.jpg' or '.jpeg' (case insensitive) with '.tif'
        self.record_full['image_path'] = self.record_full['image_path'].str.replace(r"\.jpe?g", ".tif",
                                                                                    case=False, regex=True)

        # truncating image_path column and concatenating with batch path
        self.record_full['CSV_batch'] = self.record_full['CSV_batch'].apply(
                                        lambda csv_batch: remove_before(csv_batch, "CP1"))

        self.record_full['image_path'] = self.record_full['image_path'].apply(
            lambda path_img: path.basename(path_img))

        self.record_full['image_path'] = self.path_prefix + self.record_full['CSV_batch'] + f"{os.path.sep}undatabased" + \
                                         f"{os.path.sep}" + self.record_full['image_path']


        # removing leading and trailing space from taxa

        tax_cols = ['Genus', 'Species', 'Rank 1', 'Epithet 1', 'Rank 2', 'Epithet 2']

        self.record_full[tax_cols] = self.record_full[tax_cols].map(
                                           lambda x: x.strip() if isinstance(x, str) else x)


        # filling in missing subtaxa ranks for first infraspecific rank

        self.record_full['missing_rank'] = (pd.isna(self.record_full[f'Rank 1']) & pd.notna(
                                           self.record_full[f'Epithet 1'])) | \
                                           ((self.record_full[f'Rank 1'] == '') & (self.record_full[f'Epithet 1'] != ''))

        self.record_full['missing_rank'] = self.record_full['missing_rank'].astype(bool)

        placeholder_rank = (pd.isna(self.record_full['Rank 1']) | (self.record_full['Rank 1'] == '')) & \
                           (self.record_full['missing_rank'] == True)

        # Set 'Rank 1' to 'subsp.' where the condition is True
        self.record_full.loc[placeholder_rank, 'Rank 1'] = 'subsp.'

        # parsing taxon columns
        self.record_full[['gen_spec', 'fullname',
                          'first_intra',
                          'taxname', 'hybrid_base']] = self.record_full.apply(self.taxon_concat,
                                                                              axis=1, result_type='expand')

        # setting datatypes for columns
        string_list = self.record_full.columns.to_list()

        self.record_full[string_list] = self.record_full[string_list].astype(str)

        self.record_full = self.record_full.replace(['', None, 'nan', np.nan], '')

        self.record_full = fill_empty_col(self.record_full, string_fill="[unspecified]", col_name="locality")

        self.record_full = fill_empty_col(self.record_full, string_fill="[No date on label]", col_name="verbatim_date")


    def barcode_has_record(self):
        """check if barcode / catalog number already in collectionobject table"""
        self.record_full['CatalogNumber'] = self.record_full['CatalogNumber'].apply(remove_non_numerics)
        self.record_full['CatalogNumber'] = self.record_full['CatalogNumber'].astype(str)
        self.record_full['barcode_present'] = ''

        for index, row in self.record_full.iterrows():
            barcode = row['CatalogNumber']
            barcode = barcode.zfill(9)
            sql = f'''select CatalogNumber from collectionobject
                      where CatalogNumber = {barcode};'''
            self.logger.info(f"running query: {sql}")
            db_barcode = self.specify_db_connection.get_one_record(sql)
            if db_barcode is None:
                self.record_full.loc[index, 'barcode_present'] = False
            else:
                self.record_full.loc[index, 'barcode_present'] = True

    def image_has_record(self):
        """checks if image name/barcode already in image_db"""
        self.record_full['image_present_db'] = None

        self.record_full['image_present_db'] = self.record_full['image_path'].apply(
            lambda filepath: self.image_client.check_image_db_if_file_imported(
                collection="Botany", filename=filepath, search_type='path', exact=True
            )
        )

    def check_barcode_match(self):
        """checks if filepath barcode matches catalog number barcode
            just in case merge between folder and specimen level data was not clean"""
        self.record_full['file_path_digits'] = self.record_full['image_path'].apply(
            lambda path: self.get_first_digits_from_filepath(path, field_size=9)
        )
        self.record_full['is_barcode_match'] = self.record_full.apply(lambda row: (row['file_path_digits'] ==
                                                                      row['CatalogNumber'].zfill(9)) or
                                                                      str_to_bool(row['duplicate']) is True,
                                                                      axis=1)

        self.record_full = self.record_full.drop(columns='file_path_digits')


    def check_if_images_present(self):
        """checks that each image exists, creating boolean column for later use"""

        self.record_full['image_valid'] = self.record_full.apply(
                                          lambda row: os.path.exists(f"{row['image_path']}")
                                                                        or str_to_bool(row['duplicate']) is True,
                                                                        axis=1)

    def taxon_process_row(self, row):
        """applies taxon_get to a row of the picturae python dataframe"""
        taxon_id = self.sql_csv_tools.taxon_get(
                        name=row['fulltaxon'],
                        hybrid=str_to_bool(row['Hybrid']),
                        taxname=row['taxname']
        )
        # Check for new genus to verify family assignment
        new_genus = False
        if taxon_id is None:
            genus_id = self.sql_csv_tools.taxon_get(
                name=row['Genus'],
                hybrid=str_to_bool(row['Hybrid']),
                taxname=row['taxname']
            )
            if genus_id is None:
                new_genus = True

        return taxon_id, new_genus

    def check_taxa_against_database(self):
        """check_taxa_against_database:
                concatenates every taxonomic column together to get the full taxonomic name,
                checks full taxonomic name against database and retrieves taxon_id if present
                and `None` if absent from db. In TNRS, only taxonomic names with a `None`
                result will be checked.
                args:
                    None
        """

        col_list = ['Genus', 'Species', 'Rank 1', 'Epithet 1', 'Rank 2', 'Epithet 2']

        # Build fulltaxon
        self.record_full['fulltaxon'] = (
            self.record_full[col_list].fillna('')
            .apply(lambda x: ' '.join(x[x != '']), axis=1)
            .str.strip()
        )

        # If empty or "missing taxon", fall back to Family
        self.record_full['fulltaxon'] = self.record_full.apply(
            lambda row: row['Family'] if (not row['fulltaxon'] or 'missing taxon' in row['fulltaxon'])
            else row['fulltaxon'],
            axis=1
        )

        # take the first occurrence of Hybrid/taxname/Genus per fulltaxon
        key_df = (
            self.record_full
            .loc[:, ['fulltaxon', 'Hybrid', 'taxname', 'Genus']]
            .dropna(subset=['fulltaxon'])
            .groupby('fulltaxon', as_index=False)
            .first()
        )

        # Apply your lookup exactly once per unique fulltaxon
        taxon_process_output = key_df.apply(lambda row: self.taxon_process_row(row), axis=1, result_type='expand')
        taxon_process_output.columns = ['taxon_id', 'new_genus']

        # Enforce unique index
        taxon_map_df = (
            key_df[['fulltaxon']]
            .join(taxon_process_output)
            .set_index('fulltaxon', verify_integrity=True)
        )

        # Factorize yields [0..n-1] in the order of appearance; use sort=True if you want alphabetical stability
        codes, uniques = pd.factorize(taxon_map_df.index, sort=True)
        taxon_map_df['fulltaxon_idx'] = codes  # Int64 dtype by default

        # Map the results back
        self.record_full['taxon_id'] = self.record_full['fulltaxon'].map(taxon_map_df['taxon_id'])
        self.record_full['new_genus'] = self.record_full['fulltaxon'].map(taxon_map_df['new_genus'])
        self.record_full['taxon_idx'] = self.record_full['fulltaxon'].map(taxon_map_df['fulltaxon_idx'])

        # Keep nullable Int64
        self.record_full['taxon_id'] = self.record_full['taxon_id'].astype(pd.Int64Dtype())
        self.record_full['taxon_idx'] = self.record_full['taxon_idx'].astype(pd.Int64Dtype())

        self.record_full.drop(columns=['fulltaxon'], inplace=True)

    def taxon_check_tnrs(self):
        """taxon_check_real:
           Sends the concatenated taxon column, through TNRS, to match names,
           with and without spelling mistakes, only checks base name
           for hybrids as IPNI does not work well with hybrids
           """

        bar_tax = self.record_full[pd.isna(self.record_full['taxon_id']) | (self.record_full['taxon_id'] == '')]

        if len(bar_tax) <= 0:

            self.record_full['overall_score'] = 1

            self.record_full['name_matched'] = ''

            self.record_full['matched_name_author'] = ''

        elif len(bar_tax) >= 1:

            bar_tax = bar_tax[['CatalogNumber', 'fullname']]

            resolved_taxon = iterate_taxon_resolve(bar_tax)

            resolved_taxon.fillna({'overall_score': 0}, inplace=True)

            resolved_taxon = resolved_taxon.drop(columns=["fullname", "unmatched_terms"])

            # merging columns on Catalog Number
            if len(resolved_taxon) > 0:
                self.record_full = pd.merge(self.record_full, resolved_taxon, on="CatalogNumber", how="left")
            else:
                raise ValueError("resolved TNRS data not returned")

            self.cleanup_tnrs()
        else:
            self.logger.error("bar tax length non-numeric")


    def cleanup_tnrs(self):
        """cleanup_tnrs: operations to re-consolidate rows with hybrids parsed for tnrs,
            and rows with missing rank parsed for tnrs.
            Separates qualifiers into new column as well.
            note: Threshold of .99 is set so that it will flag any taxon that differs from its match in any way,
            which is why a second taxon-concat is not run.
        """

        # re-consolidating hybrid column to fullname and removing hybrid_base column

        self.record_full['hybrid_base'] = self.record_full['hybrid_base'].astype(str).str.strip()

        hybrid_mask = (self.record_full['hybrid_base'].notna()) & (self.record_full['hybrid_base'] != '')

        self.record_full.loc[hybrid_mask, 'fullname'] = self.record_full.loc[hybrid_mask, 'hybrid_base']


        self.record_full = self.record_full.drop(columns=['hybrid_base'])

        # consolidating taxonomy with replaced rank

        self.record_full['missing_rank'] = self.record_full['missing_rank'].replace({'True': True,
                                                                                     'False': False}).astype(bool)
        # mask for successful match
        good_match = (pd.notna(self.record_full['name_matched']) & self.record_full['name_matched'] != '') & \
                     (self.record_full['overall_score'] >= .99)
        # creating mask for missing ranks
        rank_mask = (self.record_full['missing_rank'] == True) & \
                    (self.record_full['fullname'] != self.record_full['name_matched']) & good_match

        # replacing good matches with their matched names
        self.record_full.loc[rank_mask, 'fullname'] = self.record_full.loc[rank_mask, 'name_matched']

        # replacing rank for missing rank cases in first intra and full taxon
        for col in ['fullname', 'first_intra']:
            self.record_full.loc[rank_mask, col] = \
                self.record_full.loc[rank_mask, col].str.replace(" subsp. ", " var. ",
                                                                           regex=False)

        for col in ['fullname', 'gen_spec', 'first_intra', 'taxname']:
            self.record_full[col] = self.record_full[col].apply(remove_qualifiers)

        # pulling new tax IDs for corrected missing ranks

        self.record_full.loc[rank_mask, 'taxon_id'] = self.record_full.loc[rank_mask, 'fullname'].apply(
            self.sql_csv_tools.taxon_get)

        if self.tnrs_ignore is False:
            self.flag_tnrs_rows()


    def flag_tnrs_rows(self):
        """function to flag TNRS rows that do not pass the .99 match threshold"""
        taxon_to_correct = self.record_full[(self.record_full['overall_score'] < 0.99) &
                                            (pd.notna(self.record_full['overall_score'])) &
                                            (self.record_full['overall_score'] != 0)]

        try:
            taxon_correct_table = taxon_to_correct[['CSV_batch', 'fullname',
                                                    'name_matched', 'overall_score']].drop_duplicates()

            assert len(taxon_correct_table) <= 0

        except:
            raise IncorrectTaxonError(f'TNRS has rejected taxonomic names at '
                                      f'the following batches: {taxon_correct_table}')

    def read_and_merge_image_manifest(self):
        """to keep taxonomic family consistent with herbarium cabinet order,
            merges the family number from the picturae imaging manifests.
            Herbarium cabinet family number supersedes 'correct' family assignment.
        """
        batch_list = list(set(self.record_full['CSV_batch']))
        headers = ["type", "folder_barcode", "CatalogNumber", "Family", "Barcode", "Timestamp", "Path"]
        full_manifest = pd.DataFrame(columns=headers)
        # reads and concatenates each imaging manifest from the path
        for batch in batch_list:
            path_to_csv = f"{self.path_prefix}{batch}{path.sep}{batch}.csv"
            batch_manifest = pd.read_csv(path_to_csv, names=headers)
            full_manifest = pd.concat([full_manifest, batch_manifest], ignore_index=True)

        full_manifest = full_manifest.loc[full_manifest['type'].str.lower() == 'folder'.lower()]
        # keeping only family and cover barcode to merge
        full_manifest.drop(columns=["type", "CatalogNumber", "Barcode", "Timestamp", "Path"], inplace=True)

        self.record_full = pd.merge(self.record_full, full_manifest, on="folder_barcode", how="left")

        # adding boolean column for rows where manifest family number differs from accepted family and neither are NA
        self.record_full['family_diff'] = np.where(
                                          self.record_full['Family_x'].notna() & self.record_full['Family_y'].notna() &
                                          self.record_full['Family_x'].str.strip().ne("") & self.record_full['Family_y'].str.strip().ne("") &
                                          (self.record_full['Family_x'] != self.record_full['Family_y']),
                                          True, False)

        self.record_full['Family_x'] = np.where(
            self.record_full['Family_y'].notna() & self.record_full['Family_y'].str.strip().ne(""),
            self.record_full['Family_y'],
            self.record_full['Family_x']
        )

        self.record_full.drop(columns="Family_y", inplace=True)

        self.record_full.rename(columns={"Family_x": "Family"}, inplace=True)




    def write_upload_csv(self):
        """write_upload_csv: writes a copy of csv to PIC upload
            allows for manual review before uploading.
        """

        self.read_and_merge_image_manifest()

        self.record_full.drop(columns=['mostly_handwritten', 'folder_barcode', 'start_date_month',
                                       'start_date_day', 'start_date_year', 'end_date_month',
                                       'end_date_day', 'end_date_year'], inplace=True)

        file_path = f"picturae_csv{path.sep}csv_batch{path.sep}PIC_upload{path.sep}PIC_record_{self.date_range}.csv"

        #adding in blank label data field distinct from notes section


        # quoting non-numerics/non-bools to prevent punctuation from splitting columns

        cols_to_quote = self.record_full.select_dtypes(include=['object']).columns

        self.record_full[cols_to_quote] = self.record_full[cols_to_quote].astype(str)

        # replacing nas and literal string NA

        self.record_full = self.record_full.fillna(pd.NA).replace({"<NA>": pd.NA, "nan": pd.NA})

        self.record_full.to_csv(file_path, index=False, encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC)

        self.logger.info(f'DataFrame has been saved to csv as: {file_path}')


    def run_all(self):
        """run_all: runs all methods in the class in order"""
        # setting directory
        to_current_directory()
        # verifying file presence
        self.file_present()
        # merging and cleaning csv files
        self.csv_merge_and_clean()
        # renaming columns
        self.csv_colnames()
        # cleaning data
        self.col_clean()

        # check taxa against db
        self.check_taxa_against_database()

        # running taxa through TNRS
        self.taxon_check_tnrs()

        # checking if barcode record present in database
        self.barcode_has_record()

        # checking if barcode has valid image file
        self.check_if_images_present()

        # checking if image has record
        self.image_has_record()
        # checking if barcode has valid file name for barcode
        self.check_barcode_match()

        # writing csv for inspection and upload
        self.write_upload_csv()


#
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Runs checks on Picturae csvs and returns "
                                                 "wrangled csv ready for upload")

    parser.add_argument('-v', '--verbosity',
                        help='verbosity level. repeat flag for more detail',
                        default=0,
                        dest='verbose',
                        action='count')

    parser.add_argument("-t", "--tnrs_ignore", nargs="?", required=True, help="True or False, choice to "
                                                                              "ignore TNRS' matched name "
                                                                              "for taxa that score < .99")

    parser.add_argument("-ci", "--covered_ignore", nargs="?",
                        required=False, help="True or False choice to ignore warnings for covered/folded specimens",
                        default=False)

    parser.add_argument("-l", "--log_level", nargs="?",
                        default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level (default: %(default)s)")

    args = parser.parse_args()

    pic_config = get_config("Botany_PIC")

    picturae_csv_instance = CsvCreatePicturae(config=pic_config, logging_level=args.log_level,
                                              tnrs_ignore=args.tnrs_ignore, covered_ignore=args.covered_ignore)
