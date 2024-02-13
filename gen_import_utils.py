"""Docstring: This is a utility file, outlining various useful functions to be used
   for csv and image import related tasks.
"""
from datetime import datetime
import sys
import numpy as np
import pandas as pd
import json
import hmac
import settings
import os
import csv
import re
# import list tools

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


def set_slashes(path_string):
    """sets slashes in json to work on either ubuntu or windows"""
    if "\\" in path_string:
        path_string = path_string.replace('\\', f'{os.path.sep}')
    else:  # If the path contains forward slashes (Unix-like path)
        path_string = path_string.replace('/', f'{os.path.sep}')
    return path_string


def replace_slashes_in_dict(json):
    """applys the set slashes function to a json type dictionary"""
    for key, value in json.items():
        if isinstance(value, str):
            json[key] = set_slashes(value)
        elif isinstance(value, list):
            json[key] = [replace_slashes_in_dict(item) if isinstance(item, dict)
                         else set_slashes(item) for item in value]
        elif isinstance(value, dict):
            replace_slashes_in_dict(value)

def extract_last_folders(path, number: int):
    """truncates a path string to keep only the last n elements of a path"""
    path_components = path.split('/')
    return '/'.join(path_components[-number:])


def picturae_paths_list(config, date):
    """parses date arg into picturae image folder structure with prefixes"""
    date_folders = f"{int(date) // 10000}{os.path.sep}" \
                   f"{str(int(date) // 100 % 100).zfill(2)}{os.path.sep}" \
                   f"{str(int(date) % 100).zfill(2)}{os.path.sep}"

    config['PIC_SCAN_FOLDERS'] = [f"{date_folders}"]
    config['date_str'] = date
    paths = []
    for cur_dir in config['PIC_SCAN_FOLDERS']:
        full_dir = os.path.join(config['PREFIX'],
                                config['BOTANY_PREFIX'],
                                cur_dir)
        paths.append(full_dir)
        print(f"Scanning: {cur_dir}")
    return paths

def read_json_config(collection):
    """reads in json file and selects correct collection setting"""
    with open('config_collections.json') as file:
        config = json.load(file)
        config = config[collection]
        replace_slashes_in_dict(config)
    return config


def remove_two_index(value_list, column_list):
    new_value_list = []
    new_column_list = []
    for entry, column in zip(value_list, column_list):
        if isinstance(entry, float) and np.isnan(entry):
            continue

        elif pd.isna(entry):
            continue

        elif entry == '<NA>' or entry == '' or entry == 'None':
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

class InvalidFilenameError(Exception):
    pass

def get_row_value_or_default(row, column_name, default_value=None):
    """used to return row values where column may or may not be present in dataframe"""
    return row[column_name] if column_name in row else default_value

def get_first_digits_from_filepath(filepath, field_size=9):
    basename = os.path.basename(filepath)
    ints = re.findall(r'\d+', basename)
    if len(ints) == 0:
        raise InvalidFilenameError("Can't get barcode from filename")
    int_digits = int(ints[0])
    string_digits = f"{int_digits}"
    string_digits = string_digits.zfill(field_size)
    print(f"extracting digits from {filepath} to get {string_digits}")
    return string_digits