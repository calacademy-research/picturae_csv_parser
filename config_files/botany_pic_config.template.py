from os import path
sla = path.sep

# database credentials
SPECIFY_DATABASE_HOST = "db.institution.org"
SPECIFY_DATABASE_PORT = 3306
SPECIFY_DATABASE = "database"
USER = "redacted"
PASSWORD = "redacted"

COLLECTION_NAME = "Botany"

AGENT_ID = "your_agent_id"

IMAGE_SUFFIX = "[0-9]*([-_])*[0-9a-zA-Z]?.(JPG|jpg|jpeg|TIFF|tif)"

PREFIX = f"{sla}path{sla}to{sla}image{sla}folder"

PIC_SCAN_FOLDERS = f"CP1_YYYYMMDD_BATCH_0001{sla}"

DATA_FOLDER = f"csv_folder_name{sla}"
CSV_SPEC = f"{sla}specimen_csv_prefix"
CSV_FOLD = f"{sla}folder_csv_prefix"

# batch assembler terms
RESIZED_PREFIX = f'{sla}path{sla}to{sla}resized_images{sla}'

DIGILEAP_DESTINATION = f'{sla}path{sla}to{sla}digileap_folder"'

# title substrings to seperate from agent names
AGENT_FIRST_TITLES = []
AGENT_LAST_TITLES = []