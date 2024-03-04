from specify_db import SpecifyDb
import logging


class DatabaseConnectionError(Exception):
    pass

class SqlCsvTools:
    def __init__(self, config, logging_level=logging.INFO):
        self.config = config
        self.specify_db_connection = SpecifyDb(db_config_class=self.config)
        self.logger = logging.getLogger("SqlCsvTools")
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

    def get_cursor(self):
        """standard db cursor"""
        return self.specify_db_connection.get_cursor()

    def commit(self):
        """standard db commit"""
        return self.specify_db_connection.commit()


    def get_one_hybrid(self, match, fullname):
        """get_one_hybrid:
            used instead of get_one_record for hybrids to
            match multi-term hybrids irrespective of order
            args:
                match = the hybrid term of a taxonomic name e.g Genus A x B,
                        match - "A X B"
        """
        parts = match.split()
        if len(parts) == 3:
            sql = f'''SELECT TaxonID FROM taxon WHERE 
                      LOWER(FullName) LIKE "%{parts[0]}%" 
                      AND LOWER(FullName) LIKE "%{parts[1]}%"
                      AND LOWER(FullName) LIKE "%{parts[2]}%";'''

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


    def taxon_get(self, name, hybrid=False, taxname=None):

        if hybrid is False:
            result_id = self.get_one_match(tab_name="taxon", id_col="TaxonID", key_col="FullName", match=name,
                                           match_type=str)
            return result_id
        else:
            result_id = self.get_one_hybrid(match=taxname, fullname=name)

            return result_id
