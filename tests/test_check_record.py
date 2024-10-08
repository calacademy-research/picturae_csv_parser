"""tests to test the record_present, barcode_present and image_has_record functions."""
import unittest
from .pic_csv_test_class import AltCsvCreatePicturae
import pandas as pd
from .testing_tools import TestingTools

class DatabaseChecks(unittest.TestCase, TestingTools):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def setUp(self):
        """creates fake dataset with dummy columns,
          that have a small subset of representative real column names,
        """
        # initializing
        self.test_csv_create_picturae = AltCsvCreatePicturae(logging_level='DEBUG')

        # creating dummy dataset, one mistake 530923 != 530924 inserted on purpose
        # the test barcode that is set to return a false is 58719322,
        # a test path that is set to return false on image db (all db records are jpgs)
        # an unrealistically high barcode higher than digit limit in DB #
        data = {'CatalogNumber': ['530923', '98719322', '8708'],
                'image_path': ['CP1_20801212_BATCH_0001/cas0530924.jpg',
                               'CP1_20801212_BATCH_0001/cas98719322.jpg',
                               'CP1_20801212_BATCH_0001/cas0008708.tif'],
                'folder_barcode': ['2310_2', '2310_2', '2312_2'],
                'duplicate': ['False', 'False', 'False']}

        self.test_csv_create_picturae.record_full = pd.DataFrame(data)

    def test_barcode_present(self):
        """checks whether boolean column added for record present"""
        self.test_csv_create_picturae.barcode_has_record()
        # checks whether boolean column correctly added
        self.assertEqual(len(self.test_csv_create_picturae.record_full.columns), 5)
        # checks that no NAs were dropped
        self.assertEqual(len(self.test_csv_create_picturae.record_full), 3)
        # checks that the correct boolean order is returned
        test_list = list(self.test_csv_create_picturae.record_full['barcode_present'])
        self.assertEqual(test_list, [True, False, True])

    def test_if_barcode_match(self):
        """tests if there is a barcode in the barcode
           column that does not match the barcode in the img file name,
           the correct boolean is returned"""
        self.test_csv_create_picturae.check_barcode_match()
        test_list = list(self.test_csv_create_picturae.record_full['is_barcode_match'])
        self.assertEqual([False, True, True], test_list)

    def test_image_has_record(self):
        """tests if image_has_record returns true for
           one real attachment in test df"""
        self.test_csv_create_picturae.image_has_record()
        test_list = list(self.test_csv_create_picturae.record_full['image_present_db'])
        self.assertEqual([True, False, False], test_list)


    def tearDown(self):
        del self.test_csv_create_picturae