import json

import requests, hmac
import time
import sys
import server_host_settings
from uuid import uuid4
import datetime
import logging
import os
from datetime import datetime, timezone, timedelta
from urllib.parse import quote


TIME_FORMAT = "%Y-%m-%d %H:%M:%S%z"


class UploadFailureException(Exception):
    pass


class DuplicateImageException(Exception):
    pass


class DeleteFailureException(Exception):
    pass


class FileNotFoundException(Exception):
    pass


class ImageClient:
    def __init__(self, config=None):
        self.logger = logging.getLogger(f'Client.{self.__class__.__name__}')

        self.ptc_timezone = timezone(timedelta(hours=-8), name="PST")
        self.update_time_delta()
        self.config=config

    def request_with_retries(self, method, url, params=None, data=None, json=None, files=None,
                             max_duration=10800, retry_interval=0):
        """
        Makes an HTTP request with retries only on connection-related failures.

        :parameter
        - method: "GET" or "POST"
        - url: The request URL
        - max_duration: Total retry duration in seconds (default: 3 hours)
        - retry_interval: Delay between retries in seconds (default: 30 sec)
        - files: Dictionary containing file uploads

        :returns
        - Response object if successful
        - None if all retries fail
        """
        start_time = time.time()

        while True:
            last_error = ""

            # Reopen the file before each retry to prevent it from being empty
            if files:
                new_files = {key: (val[0], open(val[1].name, 'rb')) for key, val in files.items()}
            else:
                new_files = None

            try:
                if method.upper() == "GET":
                    r = requests.get(url, params=params)
                elif method.upper() == "POST":
                    r = requests.post(url, data=data, json=json, files=new_files)
                else:
                    raise ValueError("Unsupported HTTP method")

                # Close reopened files
                if new_files:
                    for f in new_files.values():
                        f[1].close()

                # If not a server error, return response
                if r.status_code not in [500, 502, 503, 504]:
                    return r

                self.logger.error(
                    f"Server error {r.status_code}: {r.text}, retrying in {retry_interval} sec..."
                )

            except (requests.ConnectionError, requests.Timeout) as e:
                last_error = e
                elapsed_time = time.time() - start_time
                if elapsed_time >= max_duration:
                    self.logger.error(f"Max retry duration reached ({max_duration}s). Giving up with last error {e}.")
                    return None

            retry_interval += retry_interval + 30

            self.logger.error(f"Connection error: {last_error}. Retrying in {retry_interval} sec...")

            time.sleep(retry_interval)

            # If URL contains "/fileupload", send a file delete request before retrying
            if "/fileupload" in url:
                self.cleanup_failed_fileupload(data=data)

    def cleanup_failed_fileupload(self, data):
        """Makes an HTTP request to clean up paths for failed /fileupload before retry
            :parameter
                data: contains the data dictionary from the prior /fileupload command
            :returns
                None
        """
        try:
            delete_data = {
                'filename': data['store'],
                'coll': data['coll'],
                'token': self.generate_token(data['store'])
            }
            delete_url = self.build_url("filedelete")
            delete_response = requests.post(url=delete_url, data=delete_data)

            if delete_response.status_code == 200:
                self.logger.info(f"File deleted at {data['store']}")
            elif delete_response.status_code >= 500:
                self.logger.error(
                    f"Server error during filedelete: {delete_response.status_code} - {delete_response.text}")
            else:
                pass
        except Exception as e:
            self.logger.error(f"Unexpected error in filedelete: {e}")


    def split_filepath(self, filepath):
        cur_filename = os.path.basename(filepath)
        cur_file_ext = cur_filename.split(".")[-1]
        return cur_filename, cur_file_ext

    def build_url(self, endpoint):
        host = server_host_settings.SERVER_NAME
        port = server_host_settings.SERVER_PORT
        prefix = server_host_settings.SERVER_PREFIX
        return f"{prefix}://{host}:{port}/{endpoint}"

    def update_time_delta_from_response(self, response):
        global server_time_delta
        try:
            timestamp = response.headers['X-Timestamp']
        except KeyError:
            server_time_delta = 0
            return

        server_time_delta = int(timestamp) - int(time.time())
        print(f"Updated server time delta to {server_time_delta}")

    def get_timestamp(self):
        """Return an integer timestamp with one second resolution for
        the current moment.
        """

        return int(time.time()) + server_time_delta

    def update_time_delta(self):
        response = requests.get(self.build_url(""))
        self.update_time_delta_from_response(response)

    def generate_token(self, filename):
        """Generate the auth token for the given filename and timestamp. """
        timestamp = self.get_timestamp()
        # print(f"image client timestamp: {timestamp}", flush=True)
        msg = str(timestamp).encode() + filename.encode()
        mac = hmac.new(server_host_settings.SERVER_KEY.encode(), msg=msg, digestmod='md5')
        return ':'.join((mac.hexdigest(), str(timestamp)))

    def delete_from_image_server(self, attach_loc, collection):
        data = {
            'filename': attach_loc,
            'coll': collection,
            'token': self.generate_token(attach_loc),
        }
        url = self.build_url("filedelete")
        self.logger.debug(f"Deleting {url} from server")
        r = self.request_with_retries(url=url, method="POST", data=data)
        if r.status_code == 404:
            raise FileNotFoundException
        if r.status_code != 200:
            self.logger.error(f"Deletion failed, aborted: {r.status_code}:{r.text}")
            raise DeleteFailureException

    def get_internal_filename(self, original_path, collection, return_list=False):
        """Retrieve the internal filename from the server based on the original file path."""
        params = {
            'file_string': quote(original_path),
            'coll': collection,
            'search_type': 'path',  # Query by file path
            'token': self.generate_token(quote(original_path))
        }

        url = self.build_url("getImageRecord")
        r = self.request_with_retries(url=url, params=params, method="GET")

        if r.status_code == 404:
            self.logger.warning(f"No record found for path {original_path} in collection {collection}.")
            return None
        elif r.status_code == 200:
            response_data = r.json()
            if isinstance(response_data, list) and len(response_data) > 0:
                if return_list:
                    internal_filenames = [entry.get("internal_filename") for entry in response_data if
                                          entry.get("internal_filename")]
                    self.logger.debug(f"Found {len(internal_filenames)} internal filenames for {original_path}.")
                    return internal_filenames
                else:
                    internal_filename = response_data[0].get("internal_filename")  # Adjust key if necessary
                    self.logger.debug(f"Found internal filename for {original_path}: {internal_filename}")
                    return internal_filename

        self.logger.error(f"Failed to retrieve internal filename for {original_path}. Status: {r.status_code}")
        return None

    def upload_to_image_server(self, full_path, redacted, collection, original_path=None, id=None):
        if full_path is None or redacted is None or collection is None:
            errstring = f"Bad input failures to upload to image server: {full_path} {redacted} {collection}"
            print(errstring, file=sys.stderr, flush=True)
            raise UploadFailureException(errstring)
        datetime_now = datetime.now(self.ptc_timezone)
        local_filename = full_path
        uuid = str(uuid4())
        extension = local_filename.split(".")[-1]
        attach_loc = uuid + "." + extension
        if original_path is not None:
            upload_path = original_path
        else:
            upload_path = full_path
        data = {
            'store': attach_loc,
            'type': 'image',
            'coll': collection,
            'token': self.generate_token(attach_loc),
            'original_filename': os.path.basename(local_filename),
            'original_path': upload_path,
            'redacted': str(redacted),
            'notes': None,
            'datetime': datetime_now.strftime(TIME_FORMAT)
        }

        url = self.build_url("fileupload")
        self.logger.debug(f"Attempting upload of local converted file {local_filename} to {url}")

        files = {'image': (attach_loc, open(local_filename, 'rb')),}

        r = self.request_with_retries(url=url, files=files, data=data, method="POST")

        if id is None:
            id = 'N/A'
        if r.status_code != 200:
            self.logger.error(f"FAIL - return code {r.status_code}. data: {data}")
            if r.status_code == 409:
                self.logger.error(f"Image already in server; skipping for {upload_path}")
                raise DuplicateImageException
            else:
                self.logger.error(f"Image upload aborted: {r.status_code}:{r.text}")

            raise UploadFailureException
        else:
            params = {
                'filename': attach_loc,
                'coll': collection,
                'type': 'image',
                'token': self.generate_token(attach_loc)
            }

            r = requests.get(self.build_url("getfileref"), params=params)
            url = r.text
            assert r.status_code == 200

            logging.info(f"Uploaded: {local_filename},{attach_loc},{url}")
            logging.info("adding to image")

        self.logger.debug("Upload to file server complete")

        return url, attach_loc

    # works for just basename +ext. "exact" does a sql "like" operation
    def check_image_db_if_filename_imported(self, collection, filename, exact=False):
        params = {
            'file_string': quote(filename),
            'coll': collection,
            'exact': exact,
            'search_type': 'filename',
            'token': self.generate_token(quote(filename))
        }

        return self.decode_response(params)

    def write_exif_image_metadata(self, exif_dict, collection, filename):

        data = {'filename': filename,
                'coll': collection,
                'exif_dict': json.dumps(exif_dict),
                'token': self.generate_token(filename)
                }

        url = self.build_url('updateexifdata')

        response = self.request_with_retries(url=url, data=data, method="POST")

        if response.status_code == 200:
            self.logger.debug(f"EXIF data for '{filename}' updated successfully.")
        else:
            self.logger.error(
                f"Failed to update EXIF data for '{filename}': {response.status_code} - {response.text}")

        self.logger.debug(f"modifying EXIF data for {filename} complete")

    def read_exif_image_data(self, collection, filename, datatype):
        params = {'filename': filename,
                  'coll': collection,
                  'dt': datatype,
                  'search_type': 'filename',
                  'token': self.generate_token(filename)
                  }
        url = self.build_url("getexifdata")

        response = self.request_with_retries(url=url, params=params, method="GET")

        if response.status_code == 200:
            return response.json()
        else:
            return None

    def decode_response(self, params):
        url = self.build_url("getImageRecord")
        r = self.request_with_retries(url=url, params=params, method="GET")
        if r.status_code == 404:
            self.logger.debug(f"Checked {params['file_string']} and found no duplicates")
            return False
        if r.status_code == 200:
            self.logger.debug(f"Checked {params['file_string']} - already imported")
            return True
        if r.status_code == 500:
            self.logger.error(f"500: Internal server error checking {params['file_string']}")
            self.logger.error(f"URL: {url}")
            assert False

        assert False

