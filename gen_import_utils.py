"""Docstring: This is a utility file, outlining various useful functions to be used
   for csv and image import related tasks.
"""
import logging
from datetime import datetime
import sys
import numpy as np
import pandas as pd
import hmac
import settings
import os
from dateutil.parser import parse

# import list tools

def fill_missing_folder_barcodes(df, spec_bar: str, fold_bar: str, parent_bar: str):
    """
    Fills in missing folder barcodes in the dataframe based on the specimen and parent barcodes.

    Args:
        df: pandas DataFrame to process.
        spec_bar: The column name containing specimen barcodes.
        fold_bar: The column name containing folder barcodes.
        parent_bar: The column name containing parent barcodes.

    Returns:
        df: The updated pandas DataFrame with filled folder barcodes.
    """
    # Create a dictionary to map SPECIMEN-BARCODE to FOLDER-BARCODE
    specimen_to_folder = df.set_index(spec_bar)[fold_bar].to_dict()

    # Create a dictionary to map PARENT-BARCODE to FOLDER-BARCODE
    parent_to_folder = df.set_index(parent_bar)[fold_bar].to_dict()

    # Fill missing FOLDER-BARCODE
    for idx, row in df.iterrows():

        # fill FOLDER-BARCODE using the SPECIMEN-BARCODE from the parent_to_folder dictionary
        if pd.isna(row[fold_bar]) and pd.notna(row[spec_bar]):
            folder_barcode = parent_to_folder.get(row[spec_bar])
            if folder_barcode:
                df.at[idx, fold_bar] = folder_barcode
        # if still empty fill using the FOLDER-BARCODE from the specimen_to_folder dictionary
        if pd.isna(row[fold_bar]) and pd.notna(row[parent_bar]):
            folder_barcode = specimen_to_folder.get(row[parent_bar])
            if folder_barcode:
                df.at[idx, fold_bar] = folder_barcode

    return df


def standardize_headers(df):
    """Convert columns to uppercase and replace spaces with hyphens to standardize alembo headers"""
    df.columns = [col.upper().replace(" ", "-") for col in df.columns]
    return df


def separate_titles(row, config):
    """used to separate list of likely titles from agent names
    """
    first_name_titles = config.AGENT_FIRST_TITLES

    last_name_titles = config.AGENT_LAST_TITLES

    # Initialize Title as empty
    row['Title'] = ''

    # Check and process first name titles
    for title in first_name_titles:
        if row['First Name'].startswith(title + " "):
            row['Title'] = title
            row['First Name'] = row['First Name'][len(title)+1:]
            break  # Assuming only one title, break after finding

    # Check and process last name titles
    for title in last_name_titles:
        if row['Last Name'].endswith(" " + title):
            row['Title'] = title
            row['Last Name'] = row['Last Name'][:-(len(title)+1)]
            break

    return row


def validate_date(date_string):
    """
    validate_date: Validates whether a date string is on the calendar, accounting for leap years.
    Is agnostic to formats between  YYYY, YYYY-MM, YYYY-MM-DD)
    Args:
        date_string: Date in string form.

    Returns:
        True if the date is valid according to its detected format; False otherwise.
    """
    if date_string and pd.notna(date_string):
        if len(date_string.split('-')[0]) != 4:
            logging.error("Year must be 4 digits.")
            return False
        try:
            parse(date_string, fuzzy=False)
            return True
        except Exception as e:
            logging.error(f"{e}")
            return False
    else:
        return True


def format_date_columns(year, month, day):
    """format_date_columns: gathers year, month, day columns
       and concatenates them into one YYYY-MM-DD date.
    """
    if not pd.isna(year) and year != "":
        date_str = f"{int(year):04d}"
        if not pd.isna(month) and month != "":
            date_str += f"-{int(month):02d}"
            if not pd.isna(day) and day != "":
                date_str += f"-{int(day):02d}"
            else:
                date_str += f"-01"
        else:
            date_str += f"-01-01"
        return date_str
    else:
        return ""


def fill_empty_col(dataframe, string_fill, col_name):
    """fill_empty_col: used to fill empty or NA entries in
       a given column with a placeholder string or NA, null etc.. """

    dataframe[col_name] = dataframe[col_name].replace(['', None, 'nan', np.nan], string_fill)

    return  dataframe


def unique_ordered_list(input_list):
    """unique_ordered_list:
            takes a list and selects only unique elements,
            while preserving order
        args:
            input_list: list which will be made to have
                        only unique elements.
    """
    unique_elements = []
    for element in input_list:
        if element not in unique_elements:
            unique_elements.append(element)
    return unique_elements


def extract_last_folders(path, number: int):
    """truncates a path string to keep only the last n elements of a path"""
    path_components = path.split('/')
    return '/'.join(path_components[-number:])


def picturae_paths_list(config):
    """parses date arg into picturae image folder structure with prefixes"""
    paths = []
    full_dir = os.path.join(config.PREFIX,
                            config.COLLECTION_PREFIX,
                            config.PIC_SCAN_FOLDERS)
    paths.append(full_dir)
    return paths

def remove_two_index(value_list, column_list):
    """if a value is NA ,NaN or None, will kick out value,
       and corresponding column name at the same index"""
    new_value_list = []
    new_column_list = []
    for entry, column in zip(value_list, column_list):
        if isinstance(entry, float) and np.isnan(entry):
            continue

        elif pd.isna(entry):
            continue

        elif entry == '<NA>' or entry == '' or entry == 'None' or \
                entry is None or entry == 'nan':
            continue

        new_value_list.append(entry)
        new_column_list.append(column)

    return new_value_list, new_column_list

# import process/directory tools
def to_current_directory():
    """to_current_directory: changes current directory to .py file location
        args:
            none
        returns:
            resets current directory to source file location
    """
    current_file_path = os.path.abspath(__file__)

    directory = os.path.dirname(current_file_path)

    os.chdir(directory)

def read_csv_file(csv_path):
    """ Helper method to read a CSV file and return the DataFrame or raise an error if file does not exist. """
    try:
        df = pd.read_csv(csv_path)
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"No csv path exists for path {csv_path}")

def get_max_subdirectory_date(parent_directory: str):
    """get_max_subdirectory_date: lists every subdirectory in a directory, presuming data is organized by date, in any
                                dash divided fomrat Y-M-D, D-M-Y etc..., pulls the largest date from the list.
                                Useful for updating config files and functions with dependent date variables
        args:
            parent_directory: the directory from which we want to list subdirectories with max date."""
    subdirect = [d for d in os.listdir(parent_directory) if os.path.isdir(os.path.join(parent_directory, d))]
    latest_date = None
    for date in subdirect:
        try:
            date = datetime.strptime(date, "%Y-%m-%d")
            if latest_date is None or date > latest_date:
                latest_date = date
        except ValueError:
            continue

    if latest_date is not None:
        return latest_date.strftime("%Y-%m-%d")
    else:
        return None

def cont_prompter():
    """cont_prompter:
            placed critical step after database checks, prompts users to
            confirm in order to continue. Allows user to check logger texts to make sure
            no unwanted data is being uploaded.
    """
    while True:
        user_input = input("Do you want to continue? (y/n): ")
        if user_input.lower() == "y":
            break
        elif user_input.lower() == "n":
            sys.exit("Script terminated by user.")
        else:
            print("Invalid input. Please enter 'y' or 'n'.")

def generate_token(timestamp, filename):
    """Generate the auth token for the given filename and timestamp.
    This is for comparing to the client submited token.
    args:
        timestamp: starting timestamp of upload batch
        file_name: the name of the datafile that was uploaded
    """
    timestamp = str(timestamp)
    if timestamp is None:
        print(f"Missing timestamp; token generation failure.")
    if filename is None:
        print(f"Missing filename, token generation failure.")
    mac = hmac.new(settings.KEY.encode(), timestamp.encode() + filename.encode(), digestmod='md5')
    print(f"Generated new token for {filename} at {timestamp}.")
    return ':'.join((mac.hexdigest(), timestamp))

def get_row_value_or_default(row, column_name, default_value=None):
    """used to return row values where column may or may not be present in dataframe"""
    return row[column_name] if column_name in row else default_value

