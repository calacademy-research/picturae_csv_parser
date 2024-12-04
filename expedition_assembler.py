"""
Docstring: The purpose of this file is to receive a CSV of random image paths and data,
and copy and assemble the following images into a new folder for export to the Zooniverse.
"""

import logging
import pandas as pd
import argparse
import os
import re
import shutil
import time
import multiprocessing
from get_configs import get_config

class AssembleExpedition:
    def __init__(self, csv_name, digileap=False):
        self.csv_name = csv_name
        self.expedition_csv = pd.read_csv(os.path.join('nfn_csv', csv_name))
        self.expedition_name = csv_name.replace(".csv", '')
        self.sep = os.path.sep
        self.config = get_config(config="Botany_PIC")
        self.resized_prefix = self.config.RESIZED_PREFIX
        if digileap:
            self.destination_path = self.config.DIGILEAP_DESTINATION
        else:
            self.destination_path = f"{self.sep}admin{self.sep}picturae_drive_mount{self.sep}CAS_for_EXP"

        # Initialize logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        self.logger.addHandler(handler)

    def duplicate_resized_images(self, batch_csv):
        """creates a copy of image for multi-barcode specimens to allow for one image per barcode."""
        for row in batch_csv.itertuples():
            if row.duplicate is True or row.duplicate == 1:
                parent_bar = row.ParentBarcode

                new_bar = row.Barcode
                # Constructing paths of new duplicate image

                old_path = row.jpg_path

                new_image_path = re.sub(r'(?<=/)\d+(?=\.jpg)', str(new_bar), old_path)

                self.logger.info(new_image_path)
                try:
                    if os.path.exists(new_image_path) is False:
                        shutil.copy2(old_path, new_image_path)
                        self.logger.info(f"Copying {old_path} to {new_image_path}")
                    else:
                        pass

                    batch_csv.loc[batch_csv['Barcode'] == new_bar, 'jpg_path'] = new_image_path

                    self.logger.info(f"Copy made of duplicate sheet {parent_bar}, at {new_bar}")

                except Exception as e:
                    raise FileNotFoundError(f"Error: {e}")
            else:
                pass

    def gather_images(self):
        """Copies image paths and CSV manifest to new expedition folder on drive."""
        # Removing sheet prefix
        self.expedition_csv['CsvBatch'] = self.expedition_csv['CsvBatch'].replace('^SHEET_', '', regex=True)

        # Combining image path prefix and jpg_path
        self.expedition_csv['CsvBatch'] = self.expedition_csv['CsvBatch'].apply(
            lambda x: "{}{}".format(self.resized_prefix, x))

        self.expedition_csv['jpg_path'] = (self.expedition_csv['CsvBatch'] + f"{self.sep}resized_jpg" +
                                           f"{self.sep}" + self.expedition_csv['jpg_path'])

        self.duplicate_resized_images(batch_csv=self.expedition_csv)

        dest_folder = os.path.join(self.destination_path, self.expedition_name)
        image_folder = os.path.join(dest_folder, 'images')
        for index, row in self.expedition_csv.iterrows():
            if not os.path.exists(image_folder):
                self.logger.info(image_folder)
                os.makedirs(image_folder)

            final_path = os.path.join(image_folder, os.path.basename(row['jpg_path']))
            if not os.path.exists(final_path):
                shutil.copy(row['jpg_path'], final_path)
                self.logger.info(f"Copying image at {row['jpg_path']} to {final_path}")
            else:
                self.logger.info(f"image already copied at destination {final_path}")

        csv_path = os.path.join(dest_folder, self.csv_name)

        # Copying over CSV manifest last
        if not os.path.exists(csv_path):
            shutil.copy(os.path.join('nfn_csv', self.csv_name), csv_path)

    def run_with_restarts(self):
        """Runs resizer using multiprocessing and will restart process on non-zero exit code to account for
        latency/rate limits on API for mounted drive."""
        self.logger.info(f"Parent process PID: {os.getpid()}")
        while True:
            process = multiprocessing.Process(target=self.gather_images)
            process.start()
            process.join()
            exit_code = process.exitcode
            if exit_code == 0:
                break
            self.logger.info(f"Script exited with code {exit_code}. Restarting in 5 minutes...")
            time.sleep(60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runs checks on Picturae CSVs and returns "
                                                 "wrangled CSV ready for upload")

    parser.add_argument('-v', '--verbosity',
                        help='verbosity level. repeat flag for more detail',
                        default=0,
                        dest='verbose',
                        action='count')

    parser.add_argument("-cn", "--csv_name", nargs="?", required=True, help="Path of CSV to process", default=None)

    parser.add_argument("-dg", "--digileap", nargs="?", required=False, help="Path of CSV to process", default=False)

    args = parser.parse_args()

    assemble_instance = AssembleExpedition(csv_name=args.csv_name, digileap=args.digileap)

    assemble_instance.run_with_restarts()
