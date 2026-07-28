import argparse
import pandas as pd
import ast
from BOT_database_updater import UpdateBotDbFields


class ImportLlama:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.record_full = pd.read_csv(csv_path)


    def parse_elevation(self, value):
        if pd.isna(value) or str(value).strip() == "":
            return pd.Series([pd.NA, pd.NA])

        try:
            elevations = ast.literal_eval(str(value))

            if not isinstance(elevations, (list, tuple)):
                elevations = [elevations]

            elevation_min = elevations[0] if len(elevations) >= 1 else pd.NA
            elevation_max = elevations[1] if len(elevations) >= 2 else pd.NA

            return pd.Series([elevation_min, elevation_max])

        except (ValueError, SyntaxError, TypeError):
            return pd.Series([pd.NA, pd.NA])


    def clean_llamaframe(self):

        # splitting elevation column
        self.record_full[["elevation_min", "elevation_max"]] = self.record_full["_elevationvalues"].apply(self.parse_elevation)



def parse_args():
    parser = argparse.ArgumentParser(
        description="Import Llama CSV results and update botany database fields."
    )

    parser.add_argument(
        "-i",
        "--csv-path",
        required=True,
        help="Path to the input CSV file.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    importer = ImportLlama(csv_path=args.csv_path)