import unittest
import pandas as pd
from unittest.mock import patch, MagicMock
import numpy as np
from .pic_csv_test_class import AltCsvCreatePicturae
from .testing_tools import TestingTools

class TestDataFlagging(unittest.TestCase, TestingTools):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setUp(self):
        # data for record_full DataFrame

        # Instance of the class where flag_missing_data resides
        self.test_csv_create_picturae = AltCsvCreatePicturae(logging_level='DEBUG')

        self.test_csv_create_picturae.record_full = pd.DataFrame({
            'Family': ['Asteraceae', 'Asteraceae', None, np.nan, 'Polemoniaceae'],
            'Rank 1': ['subsp.', None, 'var.', '', None],
            'Rank 2': [None, None, 'f.', '', ''],
            'Epithet 1': ['E1', 'E1', 'E1', 'E1', 'E1'],
            'Epithet 2': [None, 'E2', 'E2', 'E2', 'E2'],
            'Country': ['USA', None, '', None, 'Brazil'],
            'sheet_notes': ['covered', '', 'folded', '', 'Duplicate[Null]'],
            'start_date': ['2022-01-01', '2023-02-29', '1918-04-31', '2024-02-29', '2022-06-15'],
            'end_date': ['2022-12-31', '2023-12-31', '2023-01-01', '1920-05-40', '2023-06-30'],
            'CatalogNumber': ['1', '2', '3', '4', '5'],
            'CSV_batch': ['Batch1', 'Batch2', 'Batch1', 'Batch3', 'Batch2'],
            'folder_barcode': ['CP_1', 'CP_2', 'CP_3', 'CP_4', 'CP_5']
        })

    def test_missing_data_masks(self):
        missing_rank_csv, missing_family_csv,  missing_geography_csv, missing_label_csv, invalid_date_csv = \
            self.test_csv_create_picturae.missing_data_masks()

        # Test missing ranks
        self.assertEqual(len(missing_rank_csv), 3)
        self.assertTrue({'2', '4', '5'}.issubset(missing_rank_csv['CatalogNumber'].values))

        # test missing family
        self.assertEqual(len(missing_family_csv), 2)
        self.assertTrue({'3', '4'}.issubset(missing_family_csv['CatalogNumber'].values))

        # Test missing geography
        self.assertEqual(len(missing_geography_csv), 3)  # Expect 3 rows with missing geography

        # Test missing labels
        self.assertEqual(len(missing_label_csv), 2)  # Expect 2 rows with covered/folded labels

        # Test invalid dates
        self.assertEqual(len(invalid_date_csv), 3)  # Expect 3 rows with invalid dates


if __name__ == '__main__':
    unittest.main()
