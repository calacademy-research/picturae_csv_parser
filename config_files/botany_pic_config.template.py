from os import path
sla = path.sep

# database credentials
SPECIFY_DATABASE_HOST = "db.institution.org"
SPECIFY_DATABASE_PORT = 3306
SPECIFY_DATABASE = "database"
USER = "redacted"
PASSWORD = "redacted"

COLLECTION_NAME = "Botany"
REPORT_PATH = f"html_reports{sla}botany_import_monitoring.html"

AGENT_ID = "your_agent_id"

IMAGE_SUFFIX = "[0-9]*([-_])*[0-9a-zA-Z]?.(JPG|jpg|jpeg|TIFF|tif)"

PREFIX = f"{sla}path{sla}to{sla}image{sla}folder"

regex_pattern = r"(CP1_).*?(_BATCH)"

PIC_SCAN_FOLDERS = f"CP1_123456_BATCH_0001{sla}"

COLLECTION_PREFIX = f"collection{sla}folder{sla}"
DATA_FOLDER = f"csv_folder_name{sla}"
CSV_SPEC = f"{sla}specimen_csv_prefix"
CSV_FOLD = f"{sla}folder_csv_prefix"


# summary statistics, figures to configure html report
MAILING_LIST = ['email_address']

AGENT_FIRST_TITLES = []
AGENT_LAST_TITLES = []

SUMMARY_TERMS = []
SUMMARY_IMG = []
EXIF_DICT = {}