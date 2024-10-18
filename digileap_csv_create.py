"""docstring: This file is home to the CsvCreateDigi class which reads the json outputs of the
   digi-leap process and appends them into a CSV. After data checks and quality control is returned as a clean CSV
   with consistent column names for use with the botany or picturae updater."""
import logging
import pandas as pd
import json
import os
import csv  # Import to use CSV quoting options

class CsvCreateDigi:
    def __init__(self, json_path, csv_dest, logging_level=logging.INFO):
        self.logger = logging.getLogger("CsvCreateDigi")
        self.logger.setLevel(logging_level)
        self.json_path = json_path
        self.csv_dest = csv_dest
        self.digi_full = None

    def collect_keys(self):
        """Collect all unique keys from JSON, including dynamic properties."""
        main_keys = set()
        dynamic_properties_keys = set()

        with open(self.json_path, "r") as file:
            json_data = json.load(file)

        def gather_keys(d, parent_key=None):
            for key, value in d.items():
                if parent_key == "dwc:dynamicProperties":
                    dynamic_properties_keys.add(key)
                else:
                    main_keys.add(key)

                if isinstance(value, dict):
                    gather_keys(value, key)

        for item in json_data:
            gather_keys(item)

        return main_keys, dynamic_properties_keys

    def concatenate_dynamic_properties(self, dynamic_dict):
        """Concatenate dynamic property key-value pairs as a single string, separated by semicolons."""
        if not dynamic_dict:
            return ""

        properties = [f"{key}-{value}" for key, value in dynamic_dict.items()]
        return "; ".join(properties)

    def create_dataframe(self):
        """Create a DataFrame from the JSON data."""
        columns = [
            "image", "word_count", "score",
            "dwc:verbatimLocality", "dwc:habitat", "dwc:geodeticDatum",
            "dwc:verbatimCoordinates", "dwc:coordinateUncertaintyInMeters",
            "dwc:verbatimElevation", "dwc:miniumElevationInMeters",
            "dwc:maximumElevationInMeters", "dwc:dynamicProperties"
        ]

        self.digi_full = pd.DataFrame(columns=columns)

        with open(self.json_path, "r") as file:
            data = json.load(file)

        for entry in data:
            row = {}
            for col in columns:
                if col == 'dwc:dynamicProperties':
                    row[col] = self.concatenate_dynamic_properties(
                        entry.get("dwc:dynamicProperties", {})
                    )
                else:
                    row[col] = entry.get(col, None)

            self.digi_full = pd.concat([self.digi_full, pd.DataFrame([row])], ignore_index=True)

    def save_to_csv(self):
        """Save the DataFrame to a CSV file with proper quoting to handle special characters."""
        if self.digi_full is None:
            self.logger.error("DataFrame is empty. Please create the DataFrame first.")
            return

        try:
            self.digi_full.to_csv(
                self.csv_dest,
                index=False,
                sep=',',
                quoting=csv.QUOTE_ALL,
                quotechar='"',
                escapechar='\\',
                lineterminator='\n'
            )
            self.logger.info(f"CSV saved successfully to {self.csv_dest}")
        except Exception as e:
            self.logger.error(f"Failed to save CSV: {e}")

# test code

# json_path = "digileap_data/ocr_batch_1.json"
# csv_dest = f"digileap_data{os.path.sep}digi_csv{os.path.sep}CSV_ocr_batch_1.csv"
# csv_create = CsvCreateDigi(json_path=json_path, csv_dest=csv_dest, logging_level="INFO")
# csv_create.create_dataframe()
# csv_create.save_to_csv()
