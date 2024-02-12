"""picturae_csv_create: this file is for wrangling and creating the dataframe
   and csv with the parsed fields required for upload, in picturae_import.
   Uses TNRS (Taxonomic Name Resolution Service) in taxon_check/test_TNRS.R
   to catch spelling mistakes, mis-transcribed taxa.
   Source for taxon names at IPNI (International Plant Names Index): https://www.ipni.org/ """
import argparse

import pandas as pd

from taxon_parse_utils import *
from gen_import_utils import *
from string_utils import *
from os import path
from importer import Importer
from sql_csv_utils import SqlCsvTools
from specify_db import SpecifyDb
import logging
from gen_import_utils import read_json_config
from taxon_tools.BOT_TNRS import iterate_taxon_resolve
starting_time_stamp = datetime.now()


class CsvCreatePicturae(Importer):
    def __init__(self, date_string, config, logging_level):
        # self.paths = paths
        self.picturae_config = config
        self.picdb_config = read_json_config(collection="picbatch")
        super().__init__(db_config_class=self.picturae_config, collection_name="Botany")
        self.logger = logging.getLogger("CsvCreatePicturae")
        self.logger.setLevel(logging_level)
        self.init_all_vars(date_string)

        self.run_all()

    def init_all_vars(self, date_string):
        """init_all_vars:to use for testing and decluttering init function,
                            initializes all class level variables  """
        self.date_use = date_string

        self.path_prefix = picturae_paths_list(config=self.picturae_config, date=self.date_use)

        # setting up alternate db connection for batch database

        self.batch_db_connection = SpecifyDb(db_config_class=self.picdb_config)

        # setting up alternate csv tools connections

        self.sql_csv_tools = SqlCsvTools(config=self.picturae_config, logging_level=self.logger.getEffectiveLevel())

        self.batch_sql_tools = SqlCsvTools(config=self.picdb_config, logging_level=self.logger.getEffectiveLevel())


        # intializing parameters for database upload
        init_list = ['taxon_id', 'barcode',
                     'collector_number', 'collecting_event_guid',
                     'collecting_event_id',
                     'determination_guid', 'collection_ob_id', 'collection_ob_guid',
                     'name_id', 'author_sci', 'family', 'gen_spec_id', 'family_id', 'parent_author',
                     'records_dropped']

        for param in init_list:
            setattr(self, param, None)


    def file_present(self):
        """file_present:
           checks if correct filepath in working directory,
           checks if file is on input date
           checks if file folder is present.
           uses self.use_date to decide which folders to check
           args:
                none
        """

        to_current_directory()

        dir_path = self.picturae_config['DATA_FOLDER'] + f"{self.date_use}"

        dir_sub = os.path.isdir(dir_path)

        if dir_sub is True:
            folder_path = self.picturae_config['DATA_FOLDER'] + f"{self.date_use}" + \
                          self.picturae_config['CSV_FOLD'] + f"{self.date_use}" + ").csv"

            specimen_path = self.picturae_config['DATA_FOLDER'] + f"{self.date_use}" + \
                            self.picturae_config['CSV_SPEC'] + f"{self.date_use}" + ").csv"

            if os.path.exists(folder_path):
                print("Folder csv exists!")
            else:
                raise ValueError("Folder csv does not exist")

            if os.path.exists(specimen_path):
                print("Specimen csv exists!")
            else:
                raise ValueError("Specimen csv does not exist")
        else:
            raise ValueError(f"subdirectory for {self.date_use} not present")


    def csv_read_path(self, csv_level: str):
        """csv_read_path:
                reads in csv data for given date self.date_use
        args:
            folder_string: denotes whether specimen or folder level data with "folder" or "specimen"
        """

        csv_path = f'picturae_csv{path.sep}' + str(self.date_use) + f'{path.sep}picturae_' + str(csv_level) + \
                   f'({str(self.date_use)}).csv'

        import_csv = pd.read_csv(csv_path)

        return import_csv


    # note: change to merging on folder barcode once real data recieved

    def linking_csv_clean(self):
        """linking_csv_clean: function used to reformat the picturae linking csvs inta a useable format,
                                to link the specimen and folder csvs."""
        csv_path = f'picturae_csv{path.sep}' + str(self.date_use) + f'{path.sep}picturae_merge' + \
                   f'({str(self.date_use)}).csv'

        column_names = ['type', 'folder', 'specimen_barcode', 'Family', 'barcode2', 'timestamp', 'image_path']

        merge_csv = pd.read_csv(csv_path, header=None)

        merge_csv.columns = column_names

        drop_columns = ['barcode2', 'timestamp', 'image_path', 'Family', 'type']

        merge_csv.drop(columns=drop_columns, inplace=True)

        merge_csv = merge_csv[~merge_csv['barcode'].str.contains("Cover")]

        return merge_csv




    def csv_merge(self):
        """csv_merge:
                merges the folder_csv and the specimen_csv on barcode
           args:
                fold_csv: folder level csv to be input as argument for merging
                spec_csv: specimen level csv to be input as argument for merging
        """
        fold_csv = self.csv_read_path(csv_level="folder")
        spec_csv = self.csv_read_path(csv_level="specimen")

        ## uncomment when real data recieved
        # merge_csv = self.linking_csv_clean()
        # spec_csv = pd.merge(merge_csv, spec_csv, on='specimen_barcode')
        #
        # merge_len = len(spec_csv)
        #
        # spec_csv.drop_duplicates(inplace=True)
        #
        # unique_len = len(spec_csv)
        #
        # if merge_len > unique_len:
        #     raise ValueError(f"merge produced {merge_len - unique_len} duplicate records")


        # is_duplicate = spec_csv['Notes'].str.contains('\d', regex=True)
        #
        # spec_csv.loc[is_duplicate, 'CatalogNumber'] = spec_csv.loc[is_duplicate, 'Notes'].apply(remove_non_numerics)





        # checking if columns to merge contain same data
        if (set(fold_csv['specimen_barcode']) == set(spec_csv['specimen_barcode'])) is True:
            # removing duplicate columns

            common_columns = list(set(fold_csv.columns).intersection(set(spec_csv.columns)))

            common_columns.remove('specimen_barcode')

            spec_csv = spec_csv.drop(common_columns, axis=1)

            # completing merge on barcode

            self.record_full = pd.merge(fold_csv, spec_csv,
                                        on='specimen_barcode', how='inner')

            merge_len = len(self.record_full)

            self.record_full = self.record_full.drop_duplicates()

            unique_len = len(self.record_full)

            if merge_len > unique_len:
                raise ValueError(f"merge produced {merge_len-unique_len} duplicate records")

        else:
            raise ValueError("Barcode Columns do not match!")

    def csv_colnames(self):
        """csv_colnames: function to be used to rename columns to DB standards.
           args:
                none"""
        # remove columns !! review when real dataset received

        cols_drop = ['application_batch', 'csv_batch', 'object_type', 'filed_as_family',
                     'barcode_info', 'Notes', 'Feedback']

        self.record_full = self.record_full.drop(columns=cols_drop)

        # some of these are just placeholders for now

        col_dict = {'Country': 'country',
                    'State': 'state',
                    'County': 'county',
                    'specimen_barcode': 'CatalogNumber',
                    'path_jpg': 'image_path',
                    'collector_number': 'collector_number',
                    'collector_first_name 1': 'collector_first_name1',
                    'collector_middle_name 1': 'collector_middle_name1',
                    'collector_last_name 1': 'collector_last_name1',
                    'collector_first_name 2': 'collector_first_name2',
                    'collector_middle_name 2': 'collector_middle_name2',
                    'collector_last_name 2': 'collector_last_name2',
                    'collector_first_name 3': 'collector_first_name3',
                    'collector_middle_name 3': 'collector_middle_name3',
                    'collector_last_name 3': 'collector_last_name3',
                    'collector_first_name 4': 'collector_first_name4',
                    'collector_middle_name 4': 'collector_middle_name4',
                    'collector_last_name 4': 'collector_last_name4',
                    'collector_first_name 5': 'collector_first_name5',
                    'collector_middle_name 5': 'collector_middle_name5',
                    'collector_last_name 5': 'collector_last_name5',
                    'Genus': 'Genus',
                    'Species': 'Species',
                    'Qualifier': 'Qualifier',
                    'Hybrid': 'Hybrid',
                    'Taxon ID': 'RankID',
                    'Author': 'Author',
                    'Locality': 'locality',
                    'Verbatim Date': 'verbatim_date',
                    'Start Date': 'start_date',
                    'End Date': 'end_date'
                    }

        self.record_full.rename(columns=col_dict, inplace=True)



    def flag_missing_rank(self):

        missing_rank = (self.record_full['Rank 1'].isna()) & (self.record_full['Rank 2'].isna()) & \
                       (~self.record_full['Epithet 1'].isna()) & (~self.record_full['Epithet 2'].isna())

        missing_rank_csv = self.record_full.loc[missing_rank]

        if len(missing_rank_csv) > 0 :
            missing_rank_csv = missing_rank_csv['folder_barcode']

            missing_rank_csv.drop_duplicates(inplace=True)

            # self.monitoring_tools = MonitoringTools(config=self.picturae_config)
            #
            # self.monitoring_tools.send_missing_rank_report(batch_num="CP1_123456_BATCH_001",
            #                                                time_stamp=starting_time_stamp, rank_csv=missing_rank_csv)

            raise ValueError('Taxonomic name with 2 missing ranks')
        else:
            self.logger.info('missing_rank empty: No corrections needed')



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
                if pd.notna(row[index]):
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

        if not pd.isna(second_epithet_in):
            tax_name = remove_qualifiers(second_epithet_in)
        elif not pd.isna(first_epithet_in):
            tax_name = remove_qualifiers(first_epithet_in)
        elif not pd.isna(spec_in):
            tax_name = remove_qualifiers(spec_in)
        elif not pd.isna(genus_in):
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
        self.record_full['verbatim_date'] = self.record_full['verbatim_date'].apply(replace_apostrophes)
        self.record_full['locality'] = self.record_full['locality'].apply(replace_apostrophes)
        date_col_list = ['start_date', 'end_date']
        # changing date formate to Y month day
        for col_string in date_col_list:
            self.record_full[col_string] = pd.to_datetime(self.record_full[col_string],
                                                          format='%m/%d/%Y').dt.strftime('%Y-%m-%d')

        # filling in missing subtaxa ranks for first infraspecific rank

        self.record_full['missing_rank'] = pd.isna(self.record_full[f'Rank 1']) & pd.notna(
                                           self.record_full[f'Epithet 1'])

        self.record_full['missing_rank'] = self.record_full['missing_rank'].astype(bool)

        self.record_full[f'Rank 1'] = self.record_full[f'Rank 1'].fillna(
                                      self.record_full[f'Epithet 1'].apply(lambda x:
                                                                           "subsp." if pd.notna(x) else ""))

        # parsing taxon columns
        self.record_full[['gen_spec', 'fullname',
                          'first_intra',
                          'taxname', 'hybrid_base']] = self.record_full.apply(self.taxon_concat,
                                                                              axis=1, result_type='expand')

        # setting datatypes for columns
        string_list = self.record_full.columns.to_list()

        self.record_full[string_list] = self.record_full[string_list].astype(str)

        # converting hybrid column to true boolean

        self.record_full['Hybrid'] = self.record_full['Hybrid'].apply(str_to_bool)

        self.record_full = self.record_full.replace(['', None, 'nan', np.nan], pd.NA)



    def barcode_has_record(self):
        """check if barcode / catalog number already in collectionobject table"""
        self.record_full['CatalogNumber'] = self.record_full['CatalogNumber'].apply(remove_non_numerics)
        self.record_full['CatalogNumber'] = self.record_full['CatalogNumber'].astype(str)
        self.record_full['barcode_present'] = ''

        for index, row in self.record_full.iterrows():
            barcode = os.path.basename(row['CatalogNumber'])
            barcode = barcode.zfill(9)
            sql = f'''select CatalogNumber from collectionobject
                      where CatalogNumber = {barcode};'''
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
        self.record_full['is_barcode_match'] = self.record_full.apply(lambda row: row['file_path_digits'] ==
                                                                      row['CatalogNumber'].zfill(9),
                                                                      axis=1)

        self.record_full = self.record_full.drop(columns='file_path_digits')

    def check_if_images_present(self):
        """checks that each image exists, creating boolean column for later use"""

        # truncating image_path column
        self.record_full['image_path'] = self.record_full['image_path'].apply(
                                         lambda path_img: extract_last_folders(path_img, 2))

        self.record_full['image_valid'] = self.record_full['image_path'].apply(
                                          lambda path_img: os.path.exists(f"{self.path_prefix}{path.sep}{path_img}"))


    def check_taxa_against_database(self):
        """check_taxa_against_database:
                concatenates every taxonomic column together to get the full taxonomic name,
                checks full taxonomic name against database and retrieves taxon_id if present
                and `None` if absent from db. In TNRS, only taxonomic names with a `None`
                result will be checked.
                args:
                    None"""

        col_list = ['Genus', 'Species', 'Rank 1', 'Epithet 1', 'Rank 2', 'Epithet 2']

        self.record_full['fulltaxon'] = ''
        for column in col_list:
            self.record_full['fulltaxon'] += self.record_full[column].fillna('') + ' '

        self.record_full['fulltaxon'] = self.record_full['fulltaxon'].str.strip()

        self.record_full['taxon_id'] = self.record_full['fulltaxon'].apply(self.sql_csv_tools.taxon_get)

        self.record_full['taxon_id'] = self.record_full['taxon_id'].astype(pd.Int64Dtype())

        self.record_full.drop(columns=['fulltaxon'])


    def taxon_check_tnrs(self):
        """taxon_check_real:
           Sends the concatenated taxon column, through TNRS, to match names,
           with and without spelling mistakes, only checks base name
           for hybrids as IPNI does not work well with hybrids
           """

        bar_tax = self.record_full[pd.isna(self.record_full['taxon_id'])]

        bar_tax = bar_tax[['CatalogNumber', 'fullname']]

        bar_tax['taxon_id'] = bar_tax['fullname'].apply(self.sql_csv_tools.taxon_get)

        bar_tax = bar_tax.drop(columns='taxon_id')


        resolved_taxon = iterate_taxon_resolve(bar_tax)


        resolved_taxon['overall_score'].fillna(0, inplace=True)

        # filtering out taxa without exact matches , saving to db table

        unmatched_taxa = resolved_taxon[resolved_taxon["overall_score"] < .99]

        if len(unmatched_taxa) > 0:
            self.batch_sql_tools.taxon_unmatch_insert(unmatched_taxa=unmatched_taxa)

        # filtering out taxa with tnrs scores lower than .99 (basically exact match)
        resolved_taxon = resolved_taxon[resolved_taxon["overall_score"] >= .99]

        # dropping uneccessary columns

        resolved_taxon = resolved_taxon.drop(columns=["fullname", "overall_score", "unmatched_terms"])

        # merging columns on full name
        if len(resolved_taxon) > 0:
            self.record_full = pd.merge(self.record_full, resolved_taxon, on="CatalogNumber", how="left")
        else:
            self.record_full['name_matched'] = pd.NA

        upload_length = len(self.record_full.index)

        # dropping taxon rows with no match

        self.record_full = self.record_full[pd.notna(self.record_full['name_matched']) |
                                            pd.notna(self.record_full['taxon_id'])]

        self.record_full.drop(columns='taxon_id')

        clean_length = len(self.record_full.index)

        self.records_dropped = upload_length - clean_length

        if self.records_dropped > 0:
            self.logger.info(f"{self.records_dropped} rows dropped due to taxon errors")

        self.cleanup_tnrs()


    def cleanup_tnrs(self):
        """cleanup_tnrs: operations to re-consolidate rows with hybrids parsed for tnrs,
            and rows with missing rank parsed for tnrs.
            Separates qualifiers into new column as well.
        """

        # re-consolidating hybrid column to fullname and removing hybrid_base column
        hybrid_mask = self.record_full['hybrid_base'].notna()

        self.record_full.loc[hybrid_mask, 'fullname'] = self.record_full.loc[hybrid_mask, 'hybrid_base']

        self.record_full = self.record_full.drop(columns=['hybrid_base'])

        # consolidating taxonomy with replaced rank

        self.record_full['missing_rank'] = self.record_full['missing_rank'].replace({'True': True,
                                                                                     'False': False}).astype(bool)

        rank_mask = (self.record_full['missing_rank'] == True) & \
                    (self.record_full['fullname'] != self.record_full['name_matched']) & \
                    (pd.notna(self.record_full['name_matched']))

        self.record_full.loc[rank_mask, 'fullname'] = self.record_full.loc[rank_mask, 'name_matched']

        self.record_full.loc[rank_mask, 'first_intra'] = \
            self.record_full.loc[rank_mask, 'first_intra'].str.replace(" subsp. ", " var. ",
                                                                       regex=False)

        self.record_full.loc[rank_mask, 'fulltaxon'] = \
            self.record_full.loc[rank_mask, 'fulltaxon'].str.replace(" subsp. ", " var. ")

        # executing qualifier separator function
        self.record_full = separate_qualifiers(self.record_full, tax_col='fullname')

        # pulling new tax IDs for corrected missing ranks

        self.record_full.loc[rank_mask, 'taxon_id'] = self.record_full.loc[rank_mask, 'fullname'].apply(
            self.sql_csv_tools.taxon_get)


        self.record_full['gen_spec'] = self.record_full['gen_spec'].apply(remove_qualifiers)



    def write_upload_csv(self):
        """write_upload_csv: writes a copy of csv to PIC upload
            allows for manual review before uploading.
        """
        file_path = f"PIC_upload{path.sep}PIC_record_{self.date_use}.csv"

        self.record_full.to_csv(file_path, index=False)

        print(f'DataFrame has been saved to csv as: {file_path}')

    def run_all(self):
        """run_all: runs all methods in the class in order"""
        # setting directory
        to_current_directory()
        # verifying file presence
        self.file_present()
        # # modifying linking csv
        # self.linking_csv_clean()
        # merging csv files
        self.csv_merge()
        # renaming columns
        self.csv_colnames()
        # flagging taxa with multiple missing ranks
        self.flag_missing_rank()
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



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Runs checks on Picturae csvs and returns "
                                                 "wrangled csv ready for upload")
    parser.add_argument("-d", "--date",nargs="?", required=True, help="date of batch to process")
    parser.add_argument("-l", "--log_level", nargs="?",
                        default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level (default: %(default)s)")

    args = parser.parse_args()



    pic_config = read_json_config("Botany_PIC")

    picturae_csv_instance = CsvCreatePicturae(config=pic_config, date_string=args.date, logging_level=args.log_level)


# def full_run():
#     """testing function to run just the first piece o
#           f the upload process"""
#     # logger = logging.getLogger("full_run")
#     test_config = read_json_config(collection="Botany_PIC")
#
#     date_override = "20230628"
#
#     from gen_import_utils import picturae_paths_list
#
#     paths = picturae_paths_list(config=test_config, date=date_override)
#
#     CsvCreatePicturae(date_string=date_override, logging_level='DEBUG', config=test_config, paths=paths)
#
#
# full_run()
