import unittest
import shutil
import os
from .pic_csv_test_class import AltCsvCreatePicturae
from .testing_tools import TestingTools
from get_configs import get_config

class DirectoryTests(unittest.TestCase, TestingTools):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.md5_hash = self.generate_random_md5()
        self.picturae_config = get_config(config="Botany_PIC")

    """WorkingDirectoryTests: a series of unit tests to verify
        correct working directory, subdirectories."""
    def setUp(self):
        """setUP: unittest setup function creates empty csvs,
                  and folders for given test path"""
        # initializing
        self.test_csv_create_picturae = AltCsvCreatePicturae(date_string=self.md5_hash, logging_level='DEBUG')

        if self._testMethodName == "test_missing_folder_raise_error":
            pass
        else:
            # create test directories
            self.dir_path = self.picturae_config.DATA_FOLDER + f"{self.md5_hash}"

            expected_folder_path = self.dir_path + \
                                   self.picturae_config.CSV_FOLD + f"{self.md5_hash}" + "_BATCH_0001.csv"

            expected_specimen_path = self.dir_path + \
                                     self.picturae_config.CSV_SPEC + f"{self.md5_hash}" + "_BATCH_0001.csv"


            # making the directories
            os.makedirs(os.path.dirname(expected_folder_path), exist_ok=True)

            open(expected_folder_path, 'a').close()
            open(expected_specimen_path, 'a').close()



    def test_missing_folder_raise_error(self):
        """checks if incorrect sub_directory raises error from file present"""
        with self.assertRaises(ValueError) as cm:
            self.test_csv_create_picturae.file_present()
        self.assertEqual(str(cm.exception), f"subdirectory for {self.md5_hash} not present")


    def test_expected_path_date(self):
        """test_expected_path_date: tests , when the
          folders are correctly created that there is
          no exception raised"""
        try:
            self.test_csv_create_picturae.file_present()
        except Exception as e:
            self.fail(f"Exception raised: {str(e)}")

    def test_raise_specimen(self):
        """test_raise_specimen: tests whether correct value
           error is raised for missing specimen_csv"""
        # removing test path specimen
        os.remove(self.dir_path + self.picturae_config.CSV_SPEC + f"{self.md5_hash}" + "_BATCH_0001.csv")
        with self.assertRaises(ValueError) as cm:
            self.test_csv_create_picturae.file_present()
        self.assertEqual(str(cm.exception), "Specimen csv does not exist")

    def test_raise_folder(self):
        """test_raise_folder: tests whether correct value error
           is raised for missing folder_csv"""
        # removing test path folder
        os.remove(self.dir_path + self.picturae_config.CSV_FOLD + f"{self.md5_hash}" + "_BATCH_0001.csv")
        with self.assertRaises(ValueError) as cm:
            self.test_csv_create_picturae.file_present()
        self.assertEqual(str(cm.exception), "Folder csv does not exist")

    def tearDown(self):
        """destroys paths for Setup function,
           returning working directory to prior state.
           pass: for test_missing folder raise error,
           because no setup executed for that function"""

        if self._testMethodName == "test_missing_folder_raise_error":
            pass
        else:

            del self.test_csv_create_picturae
            # create test directories

            expected_folder_path = self.dir_path + \
                                   self.picturae_config.CSV_FOLD + f"{self.md5_hash}" + "_BATCH_0001.csv"

            shutil.rmtree(os.path.dirname(expected_folder_path))


if __name__ == "__main__":
    unittest.main()
