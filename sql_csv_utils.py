import traceback
import pandas as pd
from gen_import_utils import remove_two_index
import time_utils
from datetime import datetime
from datetime import timedelta
import string_utils
import sys
from specify_db import SpecifyDb
import logging
from typing import Union


class DatabaseConnectionError(Exception):
    pass


class SqlCsvTools:
    def __init__(self, config, logging_level=logging.INFO):
        self.config = config
        self.specify_db_connection = SpecifyDb(db_config_class=self.config)
        self.logger = logging.getLogger(f'Client.' + self.__class__.__name__)
        self.logger.setLevel(logging_level)
        self.check_db_connection()

    def check_db_connection(self):
        """checking whether database connection is functional"""
        try:
            self.specify_db_connection.connect()
            self.logger.info("sql_csv_tools connection established")
        except Exception as e:
            raise DatabaseConnectionError from e

    def ensure_db_connection(self):
        """Ensure that the database connection is functional. Recreate if an error is raised."""
        try:
            # Attempt to connect to the database
            self.specify_db_connection.connect()
            self.logger.info("Database connection established")
        except Exception as e:
            # If an error is raised, recreate the database connection
            self.logger.warning("Database connection error. Recreating connection...")
            self.specify_db_connection = SpecifyDb(db_config_class=self.config)
            self.specify_db_connection.connect()
            self.logger.info("Database connection recreated")

    def sql_db_connection(self):
        """standard connector"""
        return self.specify_db_connection.connect()

    def get_record(self, sql):
        """dbtools get_one_record"""
        return self.specify_db_connection.get_one_record(sql=sql)

    def get_records(self, sql):

        return self.specify_db_connection.get_records(query=sql)

    def get_cursor(self):
        """standard db cursor"""
        return self.specify_db_connection.get_cursor()

    def commit(self):
        """standard db commit"""
        return self.specify_db_connection.commit()


    # static methods
    def check_agent_name_sql(self, first_name: str, last_name: str, middle_initial: str, title: str):
        """create_name_sql: create a custom sql string, based on number of non-na arguments, the
                            database does not recognize empty strings '' and NA as equivalent.
                            Has conditional to ensure the first statement always starts with WHERE
            args:
                first_name: first name of agent
                last_name: last name of agent
                middle_initial: middle initial of agent
                title: agent's title. (mr, ms, dr. etc..)
        """
        sql = f'''SELECT AgentID FROM agent'''
        statement_count = 0
        if not pd.isna(first_name) and first_name != '':
            statement_count += 1
            sql += f''' WHERE FirstName = "{first_name}"'''
        else:
            statement_count += 1
            sql += f''' WHERE FirstName IS NULL'''

        if not pd.isna(last_name) and last_name != '':
            sql += f''' AND LastName = "{last_name}"'''

        else:
            sql += f''' AND LastName IS NULL'''

        if not pd.isna(middle_initial) and middle_initial != '':
            sql += f''' AND MiddleInitial = "{middle_initial}"'''
        else:
            sql += f''' AND MiddleInitial IS NULL'''

        if not pd.isna(title) and title != '':
            sql += f''' AND Title = "{title}"'''
        else:
            sql += f''' AND Title IS NULL'''

        sql += ''';'''

        result = self.get_record(sql=sql)

        if isinstance(result, (list, dict, set)):
            return result[0]
        else:
            return result


    def get_one_hybrid(self, match, fullname):
        """get_one_hybrid:
            used instead of get_one_record for hybrids to
            match multi-term hybrids irrespective of order
            args:
                match = the hybrid term of a taxonomic name e.g Genus A x B,
                        match - "A X B"
                fullname = the full name of the taxonomic name.
        """
        parts = match.split()
        if len(parts) == 3:
            basename = fullname.split()[0]
            sql = f'''SELECT TaxonID FROM taxon WHERE 
                      LOWER(FullName) LIKE "%{parts[0]}%" 
                      AND LOWER(FullName) LIKE "%{parts[1]}%"
                      AND LOWER(FullName) LIKE "%{parts[2]}%"
                      AND LOWER(FullName) LIKE "%{basename}%";'''

            result = self.specify_db_connection.get_records(query=sql)

            if result:
                taxon_id = result[0]
            else:
                taxon_id = None

            return taxon_id

        elif len(parts) < 3:
            taxon_id = self.get_one_match(tab_name="taxon", id_col="TaxonID", key_col="FullName", match=fullname,
                                          match_type=str)

            return taxon_id
        else:
            self.logger.error("hybrid tax name has more than 3 terms")

            return None



    def get_one_match(self, tab_name, id_col, key_col, match, match_type=str):
        """populate_sql:
                creates a custom select statement for get one record,
                from which a result can be gotten more seamlessly
                without having to rewrite the sql variable every time
           args:
                tab_name: the name of the table to select
                id_col: the name of the column in which the unique id is stored
                key_col: column on which to match values
                match: value with which to match key_col
                match_type: "string" or "integer", optional with default as "string"
                            puts quotes around sql terms or not depending on data type
        """
        sql = ""
        if match_type == str:
            sql = f'''SELECT {id_col} FROM {tab_name} WHERE `{key_col}` = "{match}";'''
        elif match_type == int:
            sql = f'''SELECT {id_col} FROM {tab_name} WHERE `{key_col}` = {match};'''

        result = self.get_record(sql=sql)

        if isinstance(result, (list, dict, set)):
            return result[0]
        else:
            return result



    def create_insert_statement(self, col_list: list, val_list: list, tab_name: str):
        """create_sql_string:
               creates a new sql insert statement given a list of db columns,
               and values to input.
            args:
                col_list: list of database table columns to fill
                val_list: list of values to input into each table
                tab_name: name of the table you wish to insert data into
        """
        # removing brackets, making sure comma is not inside of quotations
        column_list = ', '.join(col_list)
        value_list = ', '.join(f"'{value}'" if isinstance(value, str) else
                               repr(value) for value in val_list)

        sql = f'''INSERT INTO {tab_name} ({column_list}) VALUES({value_list});'''

        return sql

    def insert_table_record(self, sql):
        """create_table_record:
               general code for the inserting of a new record into any table on database,
               creates connection, and runs sql query. cursor.execute with arg multi, to
               handle multi-query commands.
           args:
               sql: the verbatim sql string, or multi sql query string to send to database
               connection: the connection parameter in the case of specify self.specify_db_connection
               logger: the logger instance of your class self.logger
               sqlite: option for sqlite configuration, as get_cursor()
                          requires database ip, which sqlite does not have
        """
        cursor = self.get_cursor()
        self.logger.debug(f"running query - {sql}")
        try:
            cursor.execute(sql)
        except Exception as e:
            print(f"Exception thrown while processing sql: {sql}\n{e}\n", flush=True)
            self.logger.error(traceback.format_exc())
        try:
            self.commit()

        except Exception as e:
            self.logger.error(f"sql debug: {e}")
            sys.exit("terminating script")

        cursor.close()

    def create_batch_record(self, start_time: datetime, end_time: datetime,
                            batch_size: int, batch_md5: str):
        """create_timestamps:
                uses starting and ending timestamps to create window for sql database purge,
                adds 10 second buffer on either end to allow sql queries to populate.
                appends each timestamp record in picturae_batch table.
            args:
                start_time: starting time stamp
                end_time: ending time stamp
        """

        end_time = end_time

        delt_time = timedelta(seconds=15)

        time_stamp_list = [start_time - delt_time, end_time + delt_time]

        column_list = ["batch_MD5",
                       "TimestampCreated",
                       "TimestampModified",
                       "StartTimeStamp",
                       "EndTimeStamp",
                       "batch_size",
                       "CreatedByAgentID",
                       "ModifiedByAgentID"
                       ]
        value_list = [f"{batch_md5}",
                      f"{time_utils.get_pst_time_now_string()}",
                      f"{time_utils.get_pst_time_now_string()}",
                      f"{time_stamp_list[0]}",
                      f"{time_stamp_list[1]}",
                      f"{batch_size}",
                      f"{self.config.AGENT_ID}",
                      f"{self.config.AGENT_ID}"
                      ]

        value_list, column_list = remove_two_index(value_list, column_list)

        sql = self.create_insert_statement(val_list=value_list, col_list=column_list, tab_name="picturae_batch")

        return sql

    def create_update_statement(self, tab_name: str, col_list: list, val_list: list, condition: str):
        """create_update_string: function used to create sql string used to upload a list of values in the database

            args:
                tab_name: name of table to update
                col_list: list of columns to update
                val_list: list of values with which to update above list of columns(order matters)
                condition: condition sql string used to select sub-sect of records to update.
        """
        val_list, col_list = remove_two_index(value_list=val_list, column_list=col_list)

        update_string = f''' SET TimestampModified = "{time_utils.get_pst_time_now_string()}", 
                            ModifiedByAgentID = "{self.config.AGENT_ID}",'''
        for index, column in enumerate(col_list):
            if isinstance(val_list[index], str):
                update_string += " " + f'''{column} = "{val_list[index]}",'''
            elif isinstance(val_list[index], float) or isinstance(val_list[index], int) or \
                    (val_list[index], type(None)):
                update_string += " " + f'''{column} = {val_list[index]},'''
            else:
                raise ValueError("unrecognized datatype for datatype parameter")

        update_string = update_string[:-1]

        sql = f'''UPDATE {tab_name}''' + update_string + ' ' + condition

        return sql

    def taxon_get(self, name, hybrid=False, taxname=None):
        """taxon_get: function to retrieve taxon id from specify database:
            args:
                name: the full taxon name to check
                hybrid: whether the taxon name belongs to a hybrid
                taxname: the name ending substring of a taxon name, only useful for retrieving hybrids.
        """
        name = name.lower()

        if hybrid is False:
            if "subsp." in name or "var." in name:
                result_id = self.get_one_match(tab_name="taxon", id_col="TaxonID", key_col="FullName", match=name,
                                               match_type=str)
                if result_id is None:
                    if "subsp." in name:
                        name = name.replace(" subsp. ", " var. ")
                    elif "var." in name:
                        name = name.replace(" var. ", " subsp. ")
                    else:
                        pass

                    result_id = self.get_one_match(tab_name="taxon", id_col="TaxonID", key_col="FullName", match=name,
                                                   match_type=str)
            else:
                result_id = self.get_one_match(tab_name="taxon", id_col="TaxonID", key_col="FullName", match=name,
                                               match_type=str)
            return result_id
        else:
            result_id = self.get_one_hybrid(match=taxname, fullname=name)

            return result_id

    def insert_taxa_added_record(self, taxon_list, df: pd.DataFrame, agent_id: Union[str, int]):
        """new_taxa_record: creates record level data for any new taxa added to the database,
                            populates useful table for qc and troubleshooting
        args:
            taxon_list: list of new taxa added to taxon tree during upload
            connection: connection instance for this sql, using self.specify_db_connection
            df: pandas dataframe, the record table uploaded to the database in question
            """
        taxa_frame = df[df['fullname'].isin(taxon_list)]
        taxa_frame = taxa_frame.drop_duplicates(subset=['fullname'])
        for index, row in taxa_frame.iterrows():
            tax_id = self.get_one_match(tab_name='picturaetaxa_added',
                                        id_col='newtaxID',
                                        key_col='fullname',
                                        match=row['fullname'],
                                        match_type=str)
            if tax_id is None:
                sql = self.create_new_tax_tab(row=row, tab_name='picturaetaxa_added', agent_id=agent_id)

                self.insert_table_record(sql=sql)

    def create_new_tax_tab(self, row, tab_name: str, agent_id: Union[str, int]):
        """create_new_tax: does a similar function as create_unmatch_tab,
                            but instead uploads a table of taxa newly added
                            to the database for QC monitoring(make sure no wonky taxa are added)
            args:
                row: row of new_taxa dataframe through which function will iterate
                df: new_taxa dataframe in order to get column index numbers
                tab_name: name of new_taxa table on mysql database.
        """
        hybrid = string_utils.str_to_bool(row['Hybrid'])

        col_list = ["fullname",
                    "TimestampCreated",
                    "TimestampModified",
                    "batch_MD5",
                    "family",
                    "name",
                    "hybrid",
                    "CreatedByAgentID",
                    "ModifiedByAgentID"]

        val_list = [f"{row['fullname']}",
                    f"{time_utils.get_pst_time_now_string()}",
                    f"{time_utils.get_pst_time_now_string()}",
                    f"{row['batch_md5']}",
                    f"{row['Family']}",
                    f"{row['taxname']}",
                    hybrid,
                    f"{agent_id}",
                    f"{agent_id}"]

        val_list, col_list = remove_two_index(val_list, col_list)

        sql = self.create_insert_statement(tab_name=tab_name, col_list=col_list,
                                           val_list=val_list)
        return sql
