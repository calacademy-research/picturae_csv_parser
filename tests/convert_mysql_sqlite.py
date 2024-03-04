import re
import argparse

def parse_command_line():
    """parsing filename from command line to convert .sql file"""
    parser = argparse.ArgumentParser(
        description=f"""
             Tool to convert mysql .sql files or data_dumps into sqlite format. 
             """,
        formatter_class=argparse.RawTextHelpFormatter, add_help=True)

    parser.add_argument('-f', '--file_path',
                        help='mysql .sql file which you want to convert to sqlite',
                        default=None)

    return parser.parse_args()


def replace_non_ascii_characters(match):
    """replaces symbols and non ascii characters with unicode"""
    char = match.group(0)
    return f'\\u{ord(char):04x}'

def convert_mysql_sqlite(filename):
    """can convert a mysql dump or .sql file into a sqlite format for use in a sqlite DB"""
    input_file = filename
    output_file = filename

    with open(input_file, 'r') as f:
        dump = f.read()
    # Replace binary '0b' with '0' and '0x' with '1'
    dump = re.sub(r"_binary '\\0'", '0', dump)
    dump = re.sub(r"_binary ''", '1', dump)
    dump = re.sub(r"\\'", "''", dump)
    dump = re.sub(r'\bbit\(1\)', 'INTEGER', dump)
    dump = re.sub(r'\bint\(11\)', 'INTEGER', dump)
    dump = re.sub(r"NOT NULL AUTO_INCREMENT", "PRIMARY KEY", dump)
    dump = re.sub(r"text DEFAULT NULL", "text", dump)
    dump = re.sub(r"ENGINE=\S+ AUTO_INCREMENT=\d+ DEFAULT CHARSET=\S+ COLLATE=\S+", "", dump)
    dump = re.sub(r"PRIMARY KEY \(`TaxonID`\),", "", dump)
    dump = re.sub(r"KEY `.*\n", "", dump)
    dump = re.sub(r"LOCK TABLES `(.*?)` WRITE;", "", dump)
    dump = re.sub(r"UNLOCK TABLES;", "", dump)

    pattern = re.compile(r'[^\x00-\x7F]')

    dump = pattern.sub(replace_non_ascii_characters, dump)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(dump)


if __name__ == '__main__':

    args = parse_command_line()

    filename = args.file_path

    convert_mysql_sqlite(filename=filename)
