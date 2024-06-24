"""Csv merge tests the csv_merge and csv_read_path functions"""
import unittest
import shutil
import os
from tests.pic_csv_test_class import AltCsvCreatePicturae
from tests.testing_tools import TestingTools
from get_configs import get_config
class CsvReadMergeTests(unittest.TestCase, TestingTools):
    """this class contains methods which test outputs of the
       csv_read_path function , and csv_merge functions from
       picturae_import.py"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.md5_hash = self.generate_random_md5()
        self.picturae_config = get_config(config="Botany_PIC")

    # will think of ways to shorten this setup function
    def setUp(self):
        """creates fake datasets with dummy columns,
          that have a small subset of representive real column names,
          so that test merges and uploads can be performed.
          """
        # print("setup called!")
        self.test_csv_create_picturae = AltCsvCreatePicturae(date_string=self.md5_hash, logging_level='DEBUG')
        # maybe create a separate function for setting up test directories

        self.dir_path = self.picturae_config.DATA_FOLDER + f"{self.md5_hash}"


        expected_folder_path = self.dir_path + \
                               self.picturae_config.CSV_FOLD + f"{self.md5_hash}" + "_BATCH_0001.csv"

        expected_specimen_path = self.dir_path + \
                                 self.picturae_config.CSV_SPEC + f"{self.md5_hash}" + "_BATCH_0001.csv"


        path_list = [expected_folder_path, expected_specimen_path]

        # making the directories
        os.makedirs(os.path.dirname(expected_folder_path), exist_ok=True)

        open(expected_folder_path, 'a').close()
        open(expected_specimen_path, 'a').close()
        # writing csvs
        self.create_fake_dataset(path_list=path_list, num_records=50)




    def test_file_empty(self):
        """tests if dataset returns as empty set or not"""
        self.assertEqual(self.test_csv_create_picturae.csv_read_path('COVER').empty, False)
        self.assertEqual(self.test_csv_create_picturae.csv_read_path('SHEET').empty, False)

    def test_file_colnumber(self):
        """tests if expected # of columns given test datasets"""
        self.assertEqual(len(self.test_csv_create_picturae.csv_read_path('COVER').columns), 11)
        self.assertEqual(len(self.test_csv_create_picturae.csv_read_path('SHEET').columns), 11)

    def test_barcode_column_present(self):
        """tests if barcode column is present
           (test if column names loaded correctly,
           specimen_barcode being in required in both csvs)"""
        self.assertEqual('SPECIMEN-BARCODE' in self.test_csv_create_picturae.csv_read_path('COVER').columns, True)
        self.assertEqual('SPECIMEN-BARCODE' in self.test_csv_create_picturae.csv_read_path('SHEET').columns, True)

    # these tests are for the csv merge function
    def test_merge_num_columns(self):
        """test merge with sample data set , checks if shared columns are removed,
           and that the merge occurs with expected # of columns"""
        # -1 as merge function drops duplicate columns automatically
        self.test_csv_create_picturae.csv_merge()
        self.assertEqual(len(self.test_csv_create_picturae.record_full.columns), 10)

    def test_index_length_matches(self):
        """checks whether dataframe, length changes,
           which would hint at barcode mismatch problem,
           folder and specimen csvs should
           always be a 100% match on barcodes
           """
        self.test_csv_create_picturae.csv_merge()
        csv_folder = self.test_csv_create_picturae.csv_read_path('SHEET')
        # test merge index before and after
        self.assertEqual(len(self.test_csv_create_picturae.record_full),
                         len(csv_folder))

    def test_unequalbarcode_raise(self):
        """tests whether the set of barcodes in the
           specimen sheet matches the set of barcodes in the merged dataframe"""
        # testing output
        self.test_csv_create_picturae.csv_merge()
        csv_specimen = self.test_csv_create_picturae.csv_read_path(csv_level="SHEET")
        self.assertEqual(set(self.test_csv_create_picturae.record_full['FOLDER-BARCODE']),
                         set(csv_specimen['FOLDER-BARCODE']))

    def test_output_isnot_empty(self):
        """tests whether merge function accidentally
           produces an empty dataframe"""
        self.test_csv_create_picturae.csv_merge()
        # testing output
        self.assertFalse(self.test_csv_create_picturae.record_full.empty)

    def tearDown(self):
        """deletes destination directories for dummy datasets"""
        # print("teardown called!")
        # deleting instance
        del self.test_csv_create_picturae
        # deleting folders

        folder_path = self.dir_path + self.picturae_config.CSV_FOLD + f"{self.md5_hash}" + "_BATCH_0001.csv"

        shutil.rmtree(os.path.dirname(folder_path))
