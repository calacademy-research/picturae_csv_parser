"""test case of the CsvCreatePicturae class which runs a reduced init method to use in unittests"""
from picturae_csv_create import CsvCreatePicturae
from gen_import_utils import read_json_config
from specify_db import SpecifyDb
import logging
from image_client import ImageClient
class AltCsvCreatePicturae(CsvCreatePicturae):
    def __init__(self, date_string, logging_level):
        self.picturae_config = read_json_config(collection="Botany_PIC")
        self.paths = ["test/path/folder"]
        self.logger = logging.getLogger("AltCsvCreatePicturae")
        self.logger.setLevel(logging_level)
        self.specify_db_connection = SpecifyDb(db_config_class=self.picturae_config)
        self.image_client = ImageClient(config=self.picturae_config)
        self.init_all_vars(date_string)

