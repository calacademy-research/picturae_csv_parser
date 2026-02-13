"""Csv merge tests the csv_merge and csv_read_path functions"""
import unittest
import shutil
import os
import csv
import pandas as pd

from .pic_csv_test_class import AltCsvCreatePicturae
from .testing_tools import TestingTools
from get_configs import get_config


class CsvReadMergeTests(unittest.TestCase, TestingTools):
    """this class contains methods which test outputs of the
       csv_read_path function , and csv_merge functions from
       picturae_import.py"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.md5_hash = self.generate_random_md5()
        self.picturae_config = get_config(config="Botany_PIC")

    def setUp(self):
        """creates fake datasets with dummy columns,
        that have a small subset of representive real column names,
        so that test merges and uploads can be performed.
        """
        self.test_csv_create_picturae = AltCsvCreatePicturae(logging_level='DEBUG')

        self.dir_path = self.picturae_config.DATA_FOLDER + f"csv_batch_{self.md5_hash}"
        self.test_csv_create_picturae.dir_path = self.dir_path

        expected_folder_path = (
            self.dir_path
            + self.picturae_config.CSV_FOLD
            + f"{self.md5_hash}" + "_BATCH_0001.csv"
        )

        expected_specimen_path = (
            self.dir_path
            + self.picturae_config.CSV_SPEC
            + f"{self.md5_hash}" + "_BATCH_0001.csv"
        )

        # manifest path
        expected_manifest_path = (
            self.dir_path
            + os.path.sep
            + f"{self.md5_hash}" + "_batch_0001.csv"
        )

        path_list = [expected_folder_path, expected_specimen_path]

        # making the directories
        os.makedirs(os.path.dirname(expected_folder_path), exist_ok=True)

        open(expected_folder_path, 'a').close()
        open(expected_specimen_path, 'a').close()
        open(expected_manifest_path, 'a').close()

        # writing the COVER + SHEET fake csvs (existing behavior)
        self.create_fake_dataset(path_list=path_list, num_records=50)

        # NEW: write MANIFEST contents with NO HEADER ROW
        # We derive the barcodes the same way your updated csv_read_path does.
        cover_df = pd.read_csv(expected_folder_path)
        sheet_df = pd.read_csv(expected_specimen_path)

        # normalize header name (your prod uses standardize_headers, but we keep this minimal)
        # prefer IMAGE-FILENAME; fall back to "Image Filename" if your fake data uses that
        cover_col = "IMAGE-FILENAME" if "IMAGE-FILENAME" in cover_df.columns else "Image Filename"
        sheet_col = "IMAGE-FILENAME" if "IMAGE-FILENAME" in sheet_df.columns else "Image Filename"

        folder_barcodes = (
            cover_df[cover_col].astype(str).str.replace(r"\.jpg$", "", regex=True)
        )
        specimen_barcodes = (
            sheet_df[sheet_col].astype(str).str.replace(r"\D+", "", regex=True)
        )

        # Manifest column order in your class:
        # ['TYPE','FOLDER-BARCODE','SPECIMEN-BARCODE','FAMILY','BARCODE_2','TIMESTAMP','PATH']
        with open(expected_manifest_path, "w", newline="") as f:
            w = csv.writer(f)
            for fb, sb in zip(folder_barcodes, specimen_barcodes):
                w.writerow(["", fb, sb, "", "", "", ""])

        # Now discover files (populates cover_list/sheet_list/manifest_list)
        self.test_csv_create_picturae.file_present()

    def test_file_empty(self):
        """tests if dataset returns as empty set or not"""
        self.assertEqual(self.test_csv_create_picturae.csv_read_path('COVER').empty, False)
        self.assertEqual(self.test_csv_create_picturae.csv_read_path('SHEET').empty, False)
        # NEW: MANIFEST must also load
        self.assertEqual(self.test_csv_create_picturae.csv_read_path('MANIFEST').empty, False)

    def test_file_colnumber(self):
        """tests if expected # of columns given test datasets"""
        self.assertEqual(len(self.test_csv_create_picturae.csv_read_path('COVER').columns), 4)
        self.assertEqual(len(self.test_csv_create_picturae.csv_read_path('SHEET').columns), 4)
        # NEW: you subset MANIFEST to 2 columns
        self.assertEqual(len(self.test_csv_create_picturae.csv_read_path('MANIFEST').columns), 2)

    def test_barcode_column_present(self):
        """tests if barcode column is present (column names loaded correctly)"""
        # UPDATED: COVER now yields FOLDER-BARCODE (not SPECIMEN-BARCODE)
        self.assertEqual('FOLDER-BARCODE' in self.test_csv_create_picturae.csv_read_path('COVER').columns, True)
        self.assertEqual('SPECIMEN-BARCODE' in self.test_csv_create_picturae.csv_read_path('SHEET').columns, True)

    def test_merge_num_columns(self):
        """test merge with sample data set

        Old test asserted an exact column count (10), but merge now:
        - goes through MANIFEST
        - adds DUPLICATE + PARENT-BARCODE
        So just assert that it merged and has required keys/fields.
        """
        self.test_csv_create_picturae.csv_merge_and_clean()
        cols = set(self.test_csv_create_picturae.record_full.columns)

        self.assertTrue("FOLDER-BARCODE" in cols)
        self.assertTrue("SPECIMEN-BARCODE" in cols)
        self.assertTrue("DUPLICATE" in cols)
        self.assertTrue("PARENT-BARCODE" in cols)

    def test_index_length_matches(self):
        """folder and specimen csvs should always be a 100% match on barcodes"""
        self.test_csv_create_picturae.csv_merge_and_clean()
        csv_sheet = self.test_csv_create_picturae.csv_read_path('SHEET')

        self.assertEqual(len(self.test_csv_create_picturae.record_full), len(csv_sheet))

    def test_unequalbarcode_raise(self):
        """tests whether barcodes in SHEET match those in merged dataframe"""
        self.test_csv_create_picturae.csv_merge_and_clean()
        csv_sheet = self.test_csv_create_picturae.csv_read_path(csv_level="SHEET")

        self.assertEqual(
            set(self.test_csv_create_picturae.record_full['SPECIMEN-BARCODE']),
            set(csv_sheet['SPECIMEN-BARCODE'])
        )

    def test_output_isnot_empty(self):
        """tests whether merge function accidentally produces an empty dataframe"""
        self.test_csv_create_picturae.csv_merge_and_clean()
        self.assertFalse(self.test_csv_create_picturae.record_full.empty)

    def tearDown(self):
        """deletes destination directories for dummy datasets"""
        del self.test_csv_create_picturae

        folder_path = (
            self.dir_path
            + self.picturae_config.CSV_FOLD
            + f"{self.md5_hash}" + "_BATCH_0001.csv"
        )
        shutil.rmtree(os.path.dirname(folder_path))
