from specify_db import SpecifyDb
import logging
from statistics import median

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

        return self.specify_db_connection.get_records(sql=sql)

    def get_cursor(self):
        """standard db cursor"""
        return self.specify_db_connection.get_cursor()

    def commit(self):
        """standard db commit"""
        return self.specify_db_connection.commit()


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
        sql = f"""
                SELECT AgentID FROM agent
                WHERE
                    (FirstName = {first_name} OR ({first_name} IS NULL AND FirstName IS NULL))
                    AND (LastName = {last_name} OR ({last_name} IS NULL AND LastName IS NULL))
                    AND (MiddleInitial = {middle_initial} OR ({middle_initial} IS NULL AND MiddleInitial IS NULL))
                    AND (Title = {title} OR ({title} IS NULL AND Title IS NULL))
            """

        result = self.get_record(sql)

        if not result:
            return None

        return result[0] if isinstance(result, tuple) else result




    def get_collecting_event_ids_by_agent_id(self, agent_id: int):
        """
        Return distinct CollectingEventID values for an AgentID from collector.
        """
        sql = f"""
            SELECT DISTINCT CollectingEventID
            FROM collector
            WHERE AgentID = {agent_id}
              AND CollectingEventID IS NOT NULL
        """
        rows = self.get_records(sql)
        return [row[0] for row in rows if row and row[0] is not None]



    def get_agent_collecting_range(self, first_name: str, last_name: str, middle_initial: str, title: str):
        """
        Given a single full-name string, find the AgentID, then look up
        CollectingEventID values in collector, then retrieve StartDate from
        collectingevent and return min / median / max 4-digit year.

        Returns a dict.
        """

        agent_id = self.check_agent_name_sql(
            first_name=first_name,
            last_name=last_name,
            middle_initial=middle_initial,
            title=title
        )

        max_year = None
        min_year = None
        median_year = None

        if agent_id is None:
            return max_year, min_year, median_year

        collecting_event_ids = self.get_collecting_event_ids_by_agent_id(agent_id)

        if not collecting_event_ids:
            return max_year, min_year, median_year

        placeholders = ", ".join(["%s"] * len(collecting_event_ids))
        sql = f"""
            SELECT YEAR(StartDate) AS StartDateYear
            FROM collectingevent
            WHERE CollectingEventID IN ({placeholders})
              AND StartDate IS NOT NULL
              AND YEAR(StartDate) IS NOT NULL
        """

        rows = self.get_records(sql)
        years = sorted(row[0] for row in rows if row and row[0] is not None)

        if not years:
            return max_year, min_year, median_year

        median_year = median(years)
        if isinstance(median_year, float) and median_year.is_integer():
            median_year = int(median_year)
            min_year = min(years)
            max_year = max(years)
            return max_year, min_year, median_year


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

            result = self.specify_db_connection.get_records(sql=sql)

            if result:
                taxon_id = result[0]
                if isinstance(taxon_id, tuple):
                    taxon_id = taxon_id[0]
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

        return self.get_record(sql=sql)

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
