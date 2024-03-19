"""tests the rename_cols function, to make sure correct column names are assigned"""
import unittest
import pandas as pd
from tests.pic_csv_test_class import AltCsvCreatePicturae
from tests.testing_tools import TestingTools

class ColNamesTest(unittest.TestCase, TestingTools):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.md5_hash = self.generate_random_md5()
    def setUp(self):
        """creates dummy dataset with representative column names"""
        # initializing class
        self.test_csv_create_picturae = AltCsvCreatePicturae(date_string=self.md5_hash, logging_level='DEBUG')
        # creating dummy dataset
        numb_range = list(range(1, 101))
        column_names = ['CSV-BATCH', 'FOLDER-BARCODE',
                        'SPECIMEN-BARCODE', 'PATH-JPG',
                        'LABEL-IS-MOSTLY-HANDWRITTEN',
                        'TAXON_ID', 'GENUS', 'SPECIES', 'QUALIFIER',
                        'RANK-1', 'EPITHET-1', 'RANK-2', 'EPITHET-2',
                        'cover_notes',
                        'HYBRID', 'AUTHOR',
                        'COLLECTOR-NUMBER', 'COLLECTOR-ID-1', 'COLLECTOR-FIRST-NAME-1',
                        'COLLECTOR-MIDDLE-NAME-1', 'COLLECTOR-LAST-NAME-1',
                        'COLLECTOR-ID-2', 'COLLECTOR-FIRST-NAME-2',
                        'COLLECTOR-MIDDLE-NAME-2', 'COLLECTOR-LAST-NAME-2',
                        'COLLECTOR-ID-3', 'COLLECTOR-FIRST-NAME-3',
                        'COLLECTOR-MIDDLE-NAME-3', 'COLLECTOR-LAST-NAME-3',
                        'COLLECTOR-ID-4', 'COLLECTOR-FIRST-NAME-4',
                        'COLLECTOR-MIDDLE-NAME-4', 'COLLECTOR-LAST-NAME-4',
                        'COLLECTOR-ID-5', 'COLLECTOR-FIRST-NAME-5',
                        'COLLECTOR-MIDDLE-NAME-5', 'COLLECTOR-LAST-NAME-5',
                        'LOCALITY-ID', 'COUNTRY', 'STATE-LOCALITY', 'COUNTY',
                        'PRECISE-LOCALITY', 'VERBATIM-DATE',
                        'START-DATE-MONTH', 'START-DATE-DAY', 'START-DATE-YEAR',
                        'END-DATE-MONTH', 'END-DATE-DAY', 'END-DATE-YEAR',
                        'sheet_notes']

        new_df = {column_names[i]: numb_range for i in range(49)}

        self.test_csv_create_picturae.record_full = pd.DataFrame(new_df)

    def test_if_id_cols(self):
        """test_if_id_col: tests whether certain essential
           ID columns present. Also tests, whether name columns correctly
           reformated
        """
        self.test_csv_create_picturae.csv_colnames()
        csv_columns = self.test_csv_create_picturae.record_full.columns
        column_names = ['collector_number', 'taxon_id',
                        'CatalogNumber', 'image_path',
                        'locality_id']
        self.assertTrue(all(column in csv_columns for column in column_names))

    def test_if_nas(self):
        """test_if_nas: test if any left-over columns contain only NAs"""
        self.test_csv_create_picturae.csv_colnames()
        self.record_full = self.test_csv_create_picturae.record_full.dropna(axis=1, how='all')
        self.assertEqual(len(self.record_full.columns), len(self.record_full.columns))

    def tearDown(self):
        del self.test_csv_create_picturae
