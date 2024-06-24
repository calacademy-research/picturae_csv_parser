"""picturae_csv_create: this file is for wrangling and creating the dataframe
   and csv with the parsed fields required for upload, in picturae_import.
   Uses TNRS (Taxonomic Name Resolution Service) in taxon_check/test_TNRS.R
   to catch spelling mistakes, mis-transcribed taxa.
   Source for taxon names at IPNI (International Plant Names Index): https://www.ipni.org/ """
import argparse
import csv
import os.path
import pandas as pd
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

class IncorrectTaxonError(Exception):
    pass

class InvalidFilenameError(Exception):
    pass


class CsvCreatePicturae:
    def __init__(self, config, tnrs_ignore, covered_ignore, logging_level, date_string=None):
        # self.paths = paths
        self.tnrs_ignore = str_to_bool(tnrs_ignore)
        self.covered_ignore = str_to_bool(covered_ignore)
        self.picturae_config = config
        self.specify_db_connection = SpecifyDb(self.picturae_config)
        self.image_client = ImageClient(config=self.picturae_config)
        self.logger = logging.getLogger("CsvCreatePicturae")
        self.logger.setLevel(logging_level)

        self.init_all_vars(date_string=date_string)

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

    def init_all_vars(self, date_string):
        """init_all_vars:to use for testing and decluttering init function,
                            initializes all class level variables  """
        self.date_use = date_string

        self.cover_list = []

        self.sheet_list = []

        self.path_prefix = self.picturae_config.PREFIX

        if self.date_use is not None:
            self.dir_path = self.picturae_config.DATA_FOLDER + f"{self.date_use}"
        else:
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
            if self.date_use is not None:
                folder_path = self.dir_path + \
                              self.picturae_config.CSV_FOLD + f"{self.date_use}" + "_BATCH_0001.csv"

                specimen_path = self.dir_path + \
                                self.picturae_config.CSV_SPEC + f"{self.date_use}" + "_BATCH_0001.csv"

                if os.path.exists(folder_path):
                    self.logger.info("Folder csv exists!")
                else:
                    raise ValueError("Folder csv does not exist")

                if os.path.exists(specimen_path):
                    self.logger.info("Specimen csv exists!")
                else:
                    raise ValueError("Specimen csv does not exist")
            else:
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
            raise ValueError(f"subdirectory for {self.date_use} not present")


    def csv_read_path(self, csv_level: str):
        """Reads in CSV data for given level and date.
        Args:
            csv_level (str): "COVER" or "SHEET" indicating the level of data.
        """
        dataframes = []

        if csv_level == "COVER":
            folder_path = self.picturae_config.CSV_FOLD
            data_list = self.cover_list
        elif csv_level == "SHEET":
            folder_path = self.picturae_config.CSV_SPEC
            data_list = self.sheet_list
        else:
            raise ValueError("Invalid csv_level value. It must be 'COVER' or 'SHEET'.")

        if self.date_use is not None:
            csv_path = self.dir_path + folder_path + f"{self.date_use}_BATCH_0001.csv"
            df = read_csv_file(csv_path)
            if " " in df.columns[0]:
                df = standardize_headers(df)
            dataframes.append(df)
        else:
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



    def drop_common_columns(self, csv: pd.DataFrame, folder=False):
        """drops columns duplicate between sheet and cover csvs"""

        drop_list = ['APPLICATION-ID', 'OBJECT-TYPE', 'APPLICATION-BATCH', 'FEEDBACK-ALEMBO', 'FEEDBACK-CALIFORNIA']
        if folder is True:
            drop_list = drop_list + ['SPECIMEN-BARCODE', 'PATH-JPG', 'CSV-BATCH']

        csv.drop(columns=drop_list, inplace=True)

        return csv


    def csv_merge(self):
        """csv_merge:
                merges the folder_csv and the specimen_csv on barcode
           args:
                fold_csv: folder level csv to be input as argument for merging
                spec_csv: specimen level csv to be input as argument for merging
        """
        fold_csv = self.csv_read_path(csv_level="COVER")
        spec_csv = self.csv_read_path(csv_level="SHEET")

        # set this way currently as
        # the folder name as it appears on the spec sheet is put into specimen barcode on cover
        fold_csv['FOLDER-BARCODE'] = fold_csv['SPECIMEN-BARCODE']

        self.drop_common_columns(fold_csv, folder=True)

        self.drop_common_columns(spec_csv)

        difference = set(fold_csv['FOLDER-BARCODE']) - set(spec_csv['FOLDER-BARCODE'])

        if len(difference) > 0:
            self.logger.warning(f"Following folder barcode not in specimen csv {difference}")


        # removing underscore suffix from barcodes
        spec_csv['SPECIMEN-BARCODE'] = spec_csv['SPECIMEN-BARCODE'].apply(remove_barcode_suffix)

        # replacing duplicate barcodes with barcodes from notes section:
        is_duplicate = spec_csv['NOTES'].astype(str).str.contains(r'\d', regex=True)

        spec_csv['DUPLICATE'] = is_duplicate

        spec_csv['PARENT-BARCODE'] = ''

        # extracting duplicate parent barcode from old image path, before replacing.
        spec_csv.loc[is_duplicate, 'PARENT-BARCODE'] = spec_csv.loc[is_duplicate, 'SPECIMEN-BARCODE']

        spec_csv.loc[is_duplicate, 'SPECIMEN-BARCODE'] = spec_csv.loc[
            is_duplicate, 'NOTES'].apply(remove_non_numerics)

        spec_csv = fill_missing_folder_barcodes(df=spec_csv, spec_bar="SPECIMEN-BARCODE",
                                                fold_bar='FOLDER-BARCODE', parent_bar="PARENT-BARCODE")

        # merging folder and specimen csvs
        self.record_full = pd.merge(fold_csv, spec_csv, on="FOLDER-BARCODE")

        # filling na
        self.record_full.fillna('', inplace=True)

        # renaming notes for sheet vs cover
        self.record_full.rename(columns={"NOTES_x": "cover_notes", "NOTES_y": "sheet_notes"}, inplace=True)

        # checking if any specimen barcodes did not match to folder barcode
        spec_difference = set(spec_csv['SPECIMEN-BARCODE']) - set(self.record_full['SPECIMEN-BARCODE'])

        spec_difference = list(spec_difference)

        spec_difference.sort(key=lambda x: int(x) if x.isdigit() else float('inf'))

        # Filter spec_csv to include only rows where SPECIMEN-BARCODE is in spec_difference
        filtered_spec_csv = spec_csv[spec_csv['SPECIMEN-BARCODE'].isin(spec_difference)]

        csv_batch_unmatch = filtered_spec_csv['CSV-BATCH'].unique()

        if len(spec_difference) > 0:
            raise ValueError(f"In the following batches {csv_batch_unmatch}, the"
                             f" following barcodes not matched to a folder {spec_difference}")


        # checking post-merge lengths

        merge_len = len(self.record_full)

        duplicates = self.record_full[self.record_full.duplicated(subset='SPECIMEN-BARCODE', keep=False)]

        unmarked_dupes = duplicates[duplicates.duplicated(subset=['SPECIMEN-BARCODE', 'COLLECTOR-NUMBER'], keep=False)==False]

        # troubleshooting code uncomment to find falsely marked duplicates
        # unmarked_all = self.record_full[
        #     self.record_full['SPECIMEN-BARCODE'].isin(unmarked_dupes['SPECIMEN-BARCODE'])]
        #
        # unmarked_all.to_csv(f'picturae_csv/csv_batch/PIC_upload/test_spec_dup.csv',
        #                           quoting=csv.QUOTE_NONNUMERIC, index=False)

        self.record_full = self.record_full.drop(unmarked_dupes.index)

        self.record_full = self.record_full.drop_duplicates()

        unique_len = len(self.record_full)

        if merge_len > unique_len:
            self.logger.error(f"Detected {merge_len-unique_len} duplicate records")



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



    def flag_missing_data(self):

        # flags in missing rank columns when > 1 infra-specific rank.

        missing_rank = (self.record_full['Rank 1'].isna() | self.record_full['Rank 1'] == '') & \
                       (self.record_full['Rank 2'].isna() | self.record_full['Rank 2'] == '') & \
                       (~self.record_full['Epithet 1'].isna() & self.record_full['Epithet 1'] != '') & \
                       (~self.record_full['Epithet 2'].isna() & self.record_full['Epithet 2'] != '')

        missing_rank_csv = self.record_full.loc[missing_rank]

        # flags if missing higher geography

        missing_geography = (self.record_full['Country'].isna() | (self.record_full['Country'] == '') |
                             (self.record_full['Country'].isnull()))

        missing_geography_csv = self.record_full.loc[missing_geography]

        # flags if label is covered or folded.

        missing_label = ["covered" in row.lower() or "folded" in row.lower() for row in self.record_full['sheet_notes']]

        missing_label_csv = self.record_full.loc[missing_label]

        if len(missing_rank_csv) > 0:
            missing_rank_set = set(missing_rank_csv['folder_barcode'])
            batch_set = set(missing_rank_csv['CSV_batch'])
            raise ValueError(f"Taxonomic names with 2 missing ranks at covers: {list(missing_rank_set)} "
                             f"in batches {batch_set}")
        else:
            self.logger.info('no missing ranks: No corrections needed')

        if len(missing_geography_csv) > 0:
            missing_geography_set = set(missing_geography_csv['CatalogNumber'])
            batch_set = set(missing_geography_csv['CSV_batch'])
            raise ValueError(f"rows missing higher geography at barcodes {missing_geography_set} "
                             f"in batches {batch_set}")
        else:
            self.logger.info('No missing higher geography: No corrections needed')

        if len(missing_label_csv) > 0:
            missing_label_set = set(missing_label_csv['CatalogNumber'])
            batch_set = set(missing_label_csv['CSV_batch'])
            raise ValueError(f"label covered or folded at barcodes {missing_label_set} "
                             f"in batches {batch_set}")
        else:
            self.logger.info('No covered or missing labels: No corrections needed')



    def taxon_concat(self, row):
        """taxon_concat:
                parses taxon columns to check taxon database, adds the Genus species, ranks, and Epithets,
                in the correct order, to create new taxon fullname in self.fullname. so that can be used for
                database checks.
            args:
                row: a row from a csv file containing taxon information with correct column names

        """
        hyb_index = self.record_full.columns.get_loc('Hybrid')
        is_hybrid = row[hyb_index]

        # defining empty strings for parsed taxon substrings
        full_name = ""
        tax_name = ""
        first_intra = ""
        gen_spec = ""
        hybrid_base = ""

        gen_index = self.record_full.columns.get_loc('Genus')
        genus = row[gen_index]

        column_sets = [
            ['Genus', 'Species', 'Rank 1', 'Epithet 1', 'Rank 2', 'Epithet 2'],
            ['Genus', 'Species', 'Rank 1', 'Epithet 1'],
            ['Genus', 'Species']
        ]

        for columns in column_sets:
            for column in columns:
                index = self.record_full.columns.get_loc(column)
                if pd.notna(row[index]) and row[index] != '':
                    if columns == column_sets[0]:
                        full_name += f" {row[index]}"
                    elif columns == column_sets[1]:
                        first_intra += f" {row[index]}"
                    elif columns == column_sets[2]:
                        gen_spec += f" {row[index]}"

        full_name = full_name.strip()
        first_intra = first_intra.strip()
        gen_spec = gen_spec.strip()
        # creating taxon name
        # creating temporary string in order to parse taxon names without qualifiers
        separate_string = remove_qualifiers(full_name)

        taxon_strings = separate_string.split()

        second_epithet_in = row[self.record_full.columns.get_loc('Epithet 2')]
        first_epithet_in = row[self.record_full.columns.get_loc('Epithet 1')]
        spec_in = row[self.record_full.columns.get_loc('Species')]
        genus_in = row[self.record_full.columns.get_loc('Genus')]
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
        # flagging taxons with multiple missing ranks
        self.flag_missing_data()
        self.record_full['verbatim_date'] = self.record_full['verbatim_date'].apply(replace_apostrophes)
        self.record_full['locality'] = self.record_full['locality'].apply(replace_apostrophes)

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

        self.record_full['image_path'] = self.record_full['CSV_batch'] + f"{os.path.sep}undatabased" + \
                                         f"{os.path.sep}" + self.record_full['image_path']

        # concatenate custom

        for col_name in list(["start", "end"]):
            self.record_full[f'{col_name}_date'] = self.record_full.apply(
                 lambda row: format_date_columns(row[f'{col_name}_date_year'],
                                                 row[f'{col_name}_date_month'], row[f'{col_name}_date_day']), axis=1)


        # removing leading and trailing space from taxa

        tax_cols = ['Genus', 'Species', 'Rank 1', 'Epithet 1', 'Rank 2', 'Epithet 2']

        self.record_full[tax_cols] = self.record_full[tax_cols].applymap(
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
        for index, row in self.record_full.iterrows():
            file_name = os.path.basename(row['image_path'])
            file_name = file_name.lower()            # file_name = file_name.rsplit(".", 1)[0]
            imported = self.image_client.check_image_db_if_filename_imported(collection="Botany",
                                                                  filename=file_name, exact=True)
            if imported is False:
                self.record_full.loc[index, 'image_present_db'] = False
            else:
                self.record_full.loc[index, 'image_present_db'] = True

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
                                          lambda row: os.path.exists(f"{self.path_prefix}{row['image_path']}")
                                                                        or str_to_bool(row['duplicate']) is True,
                                                                        axis=1)


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

        self.record_full['fulltaxon'] = ''
        # concatenating together taxonomic columns to create fulltaxon

        self.record_full['fulltaxon'] = self.record_full[col_list].fillna('').apply(lambda x: ' '.join(x[x != '']),
                                                                                    axis=1)

        self.record_full['fulltaxon'] = self.record_full['fulltaxon'].str.strip()

        # Query once per unique entry for efficiency
        unique_fulltaxons = self.record_full['fulltaxon'].unique()

        taxon_id_map = {fulltaxon: self.sql_csv_tools.taxon_get(fulltaxon) for fulltaxon in unique_fulltaxons}

        # Mapping the results back to the original DataFrame
        self.record_full['taxon_id'] = self.record_full['fulltaxon'].map(taxon_id_map)

        self.record_full['taxon_id'] = self.record_full['taxon_id'].astype(pd.Int64Dtype())

        self.record_full.drop(columns=['fulltaxon'])


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

        elif len(bar_tax) > 1:

            bar_tax = bar_tax[['CatalogNumber', 'fullname']]

            resolved_taxon = iterate_taxon_resolve(bar_tax)

            resolved_taxon['overall_score'].fillna(0, inplace=True)


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
        # mask for succesful match
        good_match = (pd.notna(self.record_full['name_matched']) & self.record_full['name_matched'] != '') & \
                     (self.record_full['overall_score'] >= .99)
        # creating mask for missing ranks
        rank_mask = (self.record_full['missing_rank'] == True) & \
                    (self.record_full['fullname'] != self.record_full['name_matched']) & good_match

        # replacing good matches with their matched names
        self.record_full.loc[rank_mask, 'fullname'] = self.record_full.loc[rank_mask, 'name_matched']

        # replacing rank for missing rank cases in first intra and full taxon

        self.record_full.loc[rank_mask, 'first_intra'] = \
            self.record_full.loc[rank_mask, 'first_intra'].str.replace(" subsp. ", " var. ",
                                                                       regex=False)

        self.record_full.loc[rank_mask, 'fulltaxon'] = \
            self.record_full.loc[rank_mask, 'fulltaxon'].str.replace(" subsp. ", " var. ",
                                                                     regex=False)
        # executing qualifier separator function
        self.record_full = separate_qualifiers(self.record_full, tax_col='fulltaxon')

        for col in ['fulltaxon', 'fullname', 'gen_spec', 'first_intra', 'taxname']:
            self.record_full[col] = self.record_full[col].apply(remove_qualifiers)

        # pulling new tax IDs for corrected missing ranks

        self.record_full.loc[rank_mask, 'taxon_id'] = self.record_full.loc[rank_mask, 'fullname'].apply(
            self.sql_csv_tools.taxon_get)

    # def fill_empty_agent:



    def write_upload_csv(self):
        """write_upload_csv: writes a copy of csv to PIC upload
            allows for manual review before uploading.
        """
        self.record_full.drop(columns=['mostly_handwritten', 'folder_barcode', 'start_date_month',
                                       'start_date_day', 'start_date_year', 'end_date_month',
                                       'end_date_day', 'end_date_year'], inplace=True)

        batch_date_list = self.record_full['CSV_batch'].apply(extract_digits, args=(8,))

        if self.date_use is not None:
            file_path = f"picturae_csv{path.sep}csv_batch{path.sep}PIC_upload{path.sep}PIC_record_{self.date_use}.csv"
        else:
            file_path = f"picturae_csv{path.sep}csv_batch{path.sep}PIC_upload{path.sep}PIC_record_{batch_date_list.min()}_{batch_date_list.max()}.csv"

        if self.tnrs_ignore is False:
            taxon_to_correct = self.record_full[(self.record_full['overall_score'] < 0.99) &
                                                (pd.notna(self.record_full['overall_score'])) &
                                                (self.record_full['overall_score'] != 0)]

            try:
                taxon_correct_list = list(taxon_to_correct['CatalogNumber'])
                assert len(taxon_correct_list) <= 0

            except:
                raise IncorrectTaxonError(f'TNRS has rejected taxonomic names at '
                                          f'the following barcodes: {taxon_correct_list}')
        else:
            pass

        self.record_full.to_csv(file_path, index=False, encoding='utf-8')

        self.logger.info(f'DataFrame has been saved to csv as: {file_path}')

    def run_all(self):
        """run_all: runs all methods in the class in order"""
        # setting directory
        to_current_directory()
        # verifying file presence
        self.file_present()
        # # modifying linking csv
        # merging csv files
        self.csv_merge()
        # renaming columns
        self.csv_colnames()
        # cleaning data
        self.col_clean()

        # checking if barcode record present in database
        self.barcode_has_record()

        # checking if barcode has valid image file
        self.check_if_images_present()

        # checking if image has record
        self.image_has_record()
        # checking if barcode has valid file name for barcode
        self.check_barcode_match()

        # check taxa against db
        self.check_taxa_against_database()

        # running taxa through TNRS
        self.taxon_check_tnrs()

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

    parser.add_argument("-d", "--date", nargs="?", required=False, help="date of batch to process", default=None)

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

    picturae_csv_instance = CsvCreatePicturae(config=pic_config, date_string=args.date, logging_level=args.log_level,
                                              tnrs_ignore=args.tnrs_ignore, covered_ignore=args.covered_ignore)

#
# def full_run():
#     """testing function to run just the first piece of the upload process"""
#     logger = logging.getLogger("full_run")
#
#     test_config = get_config(config="Botany_PIC")
#
#     date_override = None
# #
#     CsvCreatePicturae(date_string=date_override, logging_level='DEBUG', covered_ignore=True,  config=test_config,
#                       tnrs_ignore=False)
# #
# full_run()
