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

# nfn clean config terms

NFN_BLACKLIST = ["username1", "username2", "username3"]

NFN_CSV_FOLDER = "nfn_csv"

OLLAMA_URL = "http://10.1.1.1.1:11111"

# placeholder bounding elevation (6190 highest in North America)
ELEV_UPPER_BOUND = 6190

ACCEPTED_DATUMS = ["WGS", "NAD", "OSGRB", "ETRS", "ED", "GDA", "JGD", "Tokyo", "KGD",
                    "TWD", "BJS", "XAS", "GCJ", "BD", "PZ", "GTRF", "CGCS",
                    "CGS", "ITRF", "ITRS", "SAD"]

HAB_SPEC_LLM_PROMPT = ( "You are a label-cleaning AI that strictly extracts *verbatim* text for fields: `habitat`, `specimen_description` and 'locality'. "
                        "You will be given a JSON object with two or three input fields: `habitat` and `specimen_description`, and sometimes 'locality' but not always locality. "
                        "Each field may include mixed information. Your job is to remove all content that does not belong in each field according to the rules below.\n\n"
                        "📌 DO NOT add or infer any new information. ONLY retain verbatim content. If a phrase is required for sentence completeness, you may retain it even if it's borderline.\n\n"
                        "🔒 Follow these strict rules:\n\n"
                        "▶️ VERBATIM ONLY:\n"
                        "- Do not rephrase, summarize, or infer meaning.\n"
                        "- Do not turn phrases into lists or categories.\n"
                        "- Use exact phrases from the label text only.\n\n"
                        "▶️ FIELD DEFINITIONS & RULES:\n"
                        "**Habitat**:\n"
                        "- Describes the physical environment where the specimen grows.\n"
                        "- Include: substrate (e.g. 'dry sand', 'loamy soil'), associated species, vegetation type (e.g. 'open grassland'), floodplains, power lines, and life zones.\n"
                        "- General terms like 'along road', 'hills', 'on canyon slopes', or 'trail edge' go here.\n"
                        "- named Burns or fires, like 'Bean Camp burn' or 'area of Carr Fire' go here. Un-named burns can also be included\n"
                        "- Include: Associated Species and phrases like 'growing alongside' or 'associated species'"
                        "- ❌ Do NOT include place names, geographic features, road names, or phrases like 'near [named place]'. These belong to locality.\n"
                        "- ❌ Do NOT include details about the plant itself like height, color, or flowers.\n\n"
                        "**Specimen Description**:\n"
                        "- Describes the physical features of the plant only, not taxonomy or scientific name or author.\n"
                        "- Include: size, color, shape, condition, maturity, flowers, inflorescence, abundance (e.g. 'common', 'many in bloom', 'rare', 'locally common/rare' etc.), and chromosome count like 'n=14'.\n"
                        "- ❌ Do NOT include habitats or locality descriptions (e.g. 'grassland', 'near Glen Alpine').\n"
                        "- ❌ Do NOT include place names.\n\n"
                        "- ❌ Do NOT include Scientific Name or Author e.g. Eriastrum sapphirinum (Eastw.) or collomia linearis etc ...\n"
                        "**Locality**:\n"
                        "- Includes specific named places: cities, roads, parks, regions, mountain ranges, distances or bearings (e.g. '3 miles west of Jacumba').\n"
                        "- ❌ Do NOT include general environments like 'grassland', 'floodplain', or 'roadside'.\n"
                        "- ❌ Do NOT include plant traits or descriptions.\n\n"
                        "▶️ FORMATTING:\n"
                        "- Return a **valid JSON object**, like: {\"habitat\": \"...\", \"specimen_description\": \"...\", \"locality\": \"...\"}\n"
                        "- If a field is missing, return an empty string: \"field\": \"\"\n"
                        "- Do not wrap in markdown or include extra explanation.\n\n"
                        "▶️ EXAMPLES:\n"
                        "- Input: 'Dry sandy soil under sagebrush. Near McGee Creek. Small annual herb with yellow flowers.'\n"
                        "- Output: {\"habitat\": \"Dry sandy soil under sagebrush.\", \"specimen_description\": \"Small annual herb with yellow flowers.\", \"locality\": \"Near McGee Creek.\"}\n"
                        "- Input: 'Along roadside near Mono Co. border. Red flowers. Open grassland with scattered shrubs.'\n"
                        "- Output: {\"habitat\": \"Along roadside. Open grassland with scattered shrubs.\", \"specimen_description\": \"Red flowers.\", \"locality\": \"Near Mono Co. border.\"}\n"
                    )


TRS_UTM_LLM_PROMPT = ("You are a geospatial data cleaning AI that processes Township-Range-Section (TRS) and UTM coordinate fields "
    "from structured tabular data. Your job is to strictly validate and clean the following fields **verbatim**, following the rules below.\n\n"
    
    "You will receive a JSON object with one or more of the following input fields: "
    "`Township`, `Range`, `Section`, `Quadrangle`, `Utm_zone`, `Utm_easting`, `Utm_northing`, and `Datum`.\n\n"

    "Return a cleaned version of the same fields, adhering to these rules:\n\n"

    "▶️ **TRS Cleaning Rules**\n"
    "- `Township`: Match `T##N` or `T##S`, where `##` is 1–99 (leading zeros optional, 'T' optional).\n"
    "- `Range`: Match `R##E` or `R##W`, where `##` is 1–99 (leading zeros optional, 'R' optional).\n"
    "- `Section`: Match `S#` or `Sec.#` where `#` is `1–36. Leading zeros are optional.Include 'S' or 'Sec.' if present \n"
    "- `Section` may include phrases with directional headings like `NW1/4`. or `NE 1/4 of SW 1/4` etc...\n"
    "- If none of the three (Township, Range, Section) are valid, blank them all. Otherwise keep valid ones.\n"
    "- `Quadrangle`: Remove if empty or generic (e.g., 'unknown', 'n/a'). Retain if clearly named.\n\n"

    "▶️ **UTM Cleaning Rules**\n"
    "- `Utm_zone`: Valid zone is 1–60.\n"
    "- `Utm_easting`: Must be 6-digit number between 100000 and 900000.\n"
    "- `Utm_northing`: Must be 6–7 digit number, max 10,000,000.\n"
    "- `Datum`: Only retain if it matches known valid datums: NAD27, NAD83, WGS84 (case-insensitive).\n"
    "- If all three of zone, easting, and northing are invalid, blank them all.\n\n"

    "⚠️ Do not infer or guess missing parts. Only correct if the input matches the required format. "
    "Return your result as a JSON object containing only the cleaned fields.\n\n"

    "Example input:\n"
    '{\n'
    '  "Township": "t5n",\n'
    '  "Range": "r2w",\n'
    '  "Section": "Sec.03",\n'
    '  "Quadrangle": "unknown",\n'
    '  "Utm_zone": "10",\n'
    '  "Utm_easting": "489000",\n'
    '  "Utm_northing": "4225000",\n'
    '  "Datum": "nad83"\n'
    '}\n\n'

    "Example output:\n"
    '{\n'
    '  "Township": "T5N",\n'
    '  "Range": "R2W",\n'
    '  "Section": "Sec.3",\n'
    '  "Quadrangle": "",\n'
    '  "Utm_zone": "10",\n'
    '  "Utm_easting": "489000",\n'
    '  "Utm_northing": "4225000",\n'
    '  "Datum": "NAD83"\n'
    '}'
)

NFN_COL_DICT = {
            'Barcode': 'Barcode',
            'Country': 'Country',
            'State': 'State',
            'County': 'County',
            'Locality': 'Locality',
            'Genus': 'Genus',
            'Species': 'Species',
            'Epithet_1': 'Epithet_1',
            'subject_id': 'subject_id',
            'CollectorNumber': 'CollectorNumber',
            'classification_id': 'classification',
            'user_name': 'user_name',
            'T18_Herbarium Code_1': 'modifier',  # herbarium code
            'T17_Accession Number _1': 'AltCatalogNumber',  # accession number
            'T20_Habitat_1': 'Remarks',  # Habitat
            'T21_Plant Description_1': 'Text1',  # specimen description
            'T23_Elevation - Minimum_1': 'MinElevation',
            'T24_Elevation - Maximum_1': 'MaxElevation',
            'T25_Elevation Units_1': 'OriginalElevationUnit',
            'T5_Are there geographic coordinates present?_1': "coordinates_present_1",
            'T12_Township_1': 'Township_1',
            'T13_Range_1': 'Range_1',
            'T14_Section_1': 'Section_1',
            'T27_Map Quadrangle_1': 'Quadrangle_1',
            'T7_UTM Zone_1': "Utm_zone_1",
            'T9_UTM Easting_1': "Utm_easting_1",
            'T8_UTM Northing_1': "Utm_northing_1",
            'T11_UTM - Datum_1': "Utm_datum_1",
            'T4_Latitude coordinates - Verbatim_1': 'lat_verbatim_1',
            'T3_Longitude coordinates - Verbatim _1': 'long_verbatim_1',
            'T6_Datum - Latitude, Longitude_1': 'lat_long_datum_1',
            'T16_Is there another set of coordinates that need to be transcribed?_1': "coordinates_present_2",
            'T12_Township_2': 'Township_2',
            'T13_Range_2': 'Range_2',
            'T14_Section_2': 'Section_2',
            'T27_Map Quadrangle_2': 'Quadrangle_2',
            'T7_UTM Zone_2': "Utm_zone_2",
            'T9_UTM Easting_2': "Utm_easting_2",
            'T8_UTM Northing_2': "Utm_northing_2",
            'T11_UTM - Datum_2': "Utm_datum_2",
            'T4_Latitude coordinates - Verbatim_2': 'lat_verbatim_2',
            'T3_Longitude coordinates - Verbatim _2': 'long_verbatim_2',
            'T6_Datum - Latitude, Longitude_2': 'lat_long_datum_2',
            'T16_Is there another set of coordinates that need to be transcribed?_2': "coordinates_present_3",
            'T12_Township_3': 'Township_3',
            'T13_Range_3': 'Range_3',
            'T14_Section_3': 'Section_3',
            'T27_Map Quadrangle_3': 'Quadrangle_3',
            'T7_UTM Zone_3': "Utm_zone_3",
            'T9_UTM Easting_3': "Utm_easting_3",
            'T8_UTM Northing_3': "Utm_northing_3",
            'T11_UTM - Datum_3': "Utm_datum_3",
            'T4_Latitude coordinates - Verbatim_3': 'lat_verbatim_3',
            'T3_Longitude coordinates - Verbatim _3': 'long_verbatim_3',
            'T6_Datum - Latitude, Longitude_3': 'lat_long_datum_3'
        }