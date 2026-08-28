import os.path
import unittest
import pandas as pd
import random
from string_utils import remove_non_numerics
from .pic_csv_test_class import AltCsvCreatePicturae
from .testing_tools import TestingTools


class DatabaseChecks(unittest.TestCase, TestingTools):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setUp(self):
        """Dynamically generate one real image path and two invalid paths for testing."""
        self.test_csv_create_picturae = AltCsvCreatePicturae(logging_level='DEBUG')

        # Get a random real record from the database
        sql = '''SELECT origFilename FROM attachment WHERE origFilename is NOT NULL and 
                 origFilename like "%/%" ORDER BY TimestampCreated asc limit 1'''

        real_record = self.test_csv_create_picturae.specify_db_connection.get_one_record(sql=sql)

        real_catalog = str(int(remove_non_numerics(os.path.basename(real_record))) + 1)

        real_path = str(real_record)

        # Generate fake barcode 1 (very large number)
        fake_catalog_1 = str(random.randint(30000000, 99999999))
        fake_path_1 = f"CP1_FAKE_BATCH/cas{fake_catalog_1}.jpg"

        # Generate fake barcode 2 (includes letters)
        fake_catalog_2 = '00000000'
        fake_path_2 = f"CP1_FAKE_BATCH/cas{fake_catalog_2}.tif"

        data = {
            'CatalogNumber': [real_catalog, fake_catalog_1, fake_catalog_2],
            'image_path': [real_path, fake_path_1, fake_path_2],
            'folder_barcode': ['2310_2', '2310_2', '2312_2'],
            'duplicate': ['False', 'False', 'False']
        }

        self.test_csv_create_picturae.record_full = pd.DataFrame(data)

    def test_barcode_present(self):
        self.test_csv_create_picturae.barcode_has_record()
        self.assertEqual(len(self.test_csv_create_picturae.record_full.columns), 5)
        self.assertEqual(len(self.test_csv_create_picturae.record_full), 3)
        self.assertEqual(
            list(self.test_csv_create_picturae.record_full['barcode_present']),
            [True, False, False]
        )

    def test_if_barcode_match(self):
        self.test_csv_create_picturae.check_barcode_match()
        test_list = list(self.test_csv_create_picturae.record_full['is_barcode_match'])
        self.assertEqual(test_list, [False, True, True]) # real is offset by 1 should fail

    def test_image_has_record(self):
        self.test_csv_create_picturae.image_has_record()
        test_list = list(self.test_csv_create_picturae.record_full['image_present_db'])
        self.assertEqual(test_list, [True, False, False])

    def tearDown(self):
        del self.test_csv_create_picturae
