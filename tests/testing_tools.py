"""base class for testing containing useful functions for Setup and Teardown"""
from faker import Faker
import csv
from PIL import Image
import os
import hashlib
import random
import string


class TestingTools:

    def create_fake_dataset(self, num_records: int, path_list: list):
        fake = Faker()

        for path in path_list:
            lower = path.lower()
            is_cover = "cover" in lower
            is_sheet = "sheet" in lower

            with open(path, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)

                col_list = [
                    "IMAGE-FILENAME",
                    "FOLDER-BARCODE",
                    "SPECIMEN-BARCODE",
                    "NOTES",
                    "DDD-BATCH-NAME",
                    "PICTURAE-BATCH-NAME",
                ]

                # For COVER/SHEET, only keep the non-barcode fields.
                # Manifest keeps barcode columns.
                if is_cover or is_sheet:
                    to_remove = {"FOLDER-BARCODE", "SPECIMEN-BARCODE"}
                    col_list = [c for c in col_list if c not in to_remove]

                writer.writerow(col_list)

                ordered_bar = 123456  # stable base

                for i in range(num_records):
                    digits = f"{ordered_bar + i}" + ".jpg"
                    folder_barcode = f"Cover{digits}" + "_2"
                    specimen_barcode = f"{ordered_bar + i}"

                    if is_cover:
                        # COVER: IMAGE-FILENAME ends with .jpg; you strip .jpg to derive folder barcode
                        image_filename = f"{folder_barcode}.jpg"
                        notes = "cover note"
                    elif is_sheet:
                        # SHEET: IMAGE-FILENAME has non-numerics; you strip to digits for specimen barcode
                        image_filename = f"{digits}"
                        notes = "sheet note"
                    else:
                        # MANIFEST (or anything else): include both barcodes; IMAGE-FILENAME can be blank
                        image_filename = ""
                        notes = "manifest note"

                    ddd_batch = f"DDD_{self.generate_random_md5()[:6]}"
                    picturae_batch = f"{self.generate_random_md5()[:8]}_BATCH_0001"

                    # Build row values matching the header order
                    row_map = {
                        "IMAGE-FILENAME": f"{image_filename}".strip(),
                        "FOLDER-BARCODE": f"{folder_barcode}".strip(),
                        "SPECIMEN-BARCODE": f"{specimen_barcode}".strip(),
                        "NOTES": notes,
                        "DDD-BATCH-NAME": ddd_batch,
                        "PICTURAE-BATCH-NAME": picturae_batch,
                    }
                    writer.writerow([row_map[c] for c in col_list])

            print(f"Fake dataset {path} with {num_records} records created successfully")

    def create_test_images(self, barcode_list: list, color: str, expected_dir: str):
        """create_test_images:
                creates a number of standard test images in a range of barcodes,
                and with a specific date string
           args:
                barcode_list: a list or range() of barcodes that
                              you wish to create dummy images for.
                date_string: a date string , with which to name directory
                             in which to create and store the dummy images
        """
        image = Image.new('RGB', (1000, 1000), color=color)

        barcode_list = barcode_list
        for barcode in barcode_list:
            expected_image_path = expected_dir + f"/{barcode}.tif"
            os.makedirs(os.path.dirname(expected_image_path), exist_ok=True)
            print(f"Created directory: {os.path.dirname(expected_image_path)}")
            image.save(expected_image_path)

    def generate_random_md5(self):
        """generate_random_md5: creates random combination of characters and digits, to create
                                a unique md5 code."""
        characters = string.ascii_letters + string.digits
        random_string = ''.join(random.choice(characters) for _ in range(16))
        md5_hash = hashlib.md5(random_string.encode()).hexdigest()

        return md5_hash
