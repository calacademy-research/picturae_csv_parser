import traceback
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from statistics import median
from typing import Union, Optional, List, Sequence, Any

import pandas as pd

from specify_db import SpecifyDb
from gen_import_utils import remove_two_index
import time_utils
import string_utils


class DatabaseConnectionError(Exception):
    pass


@dataclass
class SqlStatement:
    sql: str
    params: Optional[Sequence[Any]] = None


class SqlCsvTools:
    def __init__(self, config, logging_level=logging.INFO):
        self.config = config
        self.specify_db_connection = SpecifyDb(db_config_class=self.config)
        self.logger = logging.getLogger(f"Client.{self.__class__.__name__}")
        self.logger.setLevel(logging_level)
        self.check_db_connection()

    def check_db_connection(self):
        """Check whether database connection is functional."""
        try:
            self.specify_db_connection.connect()
            self.logger.info("sql_csv_tools connection established")
        except Exception as e:
            raise DatabaseConnectionError from e

    def ensure_db_connection(self):
        """Ensure that the database connection is functional. Recreate if needed."""
        try:
            self.specify_db_connection.connect()
            self.logger.info("Database connection established")
        except Exception:
            self.logger.warning("Database connection error. Recreating connection...")
            self.specify_db_connection = SpecifyDb(db_config_class=self.config)
            self.specify_db_connection.connect()
            self.logger.info("Database connection recreated")

    def sql_db_connection(self):
        """Standard connector."""
        return self.specify_db_connection.connect()

    def get_record(self, sql, params=None):
        """Get one record."""
        return self.specify_db_connection.get_one_record(sql=sql, params=params)

    def get_records(self, sql, params=None):
        """Get many records."""
        return self.specify_db_connection.get_records(sql=sql, params=params)

    def get_cursor(self, buffered=False):
        """Standard db cursor."""
        return self.specify_db_connection.get_cursor(buffered=buffered)

    def commit(self):
        """Standard db commit."""
        return self.specify_db_connection.commit()

    @staticmethod
    def _normalize_nullable(value):
        """Convert empty strings to None for SQL NULL-safe matching."""
        if value is None:
            return None
        if isinstance(value, str) and value.strip() == "":
            return None
        return value

    @staticmethod
    def _extract_scalar(result):
        """Normalize get_one_record/get_records style outputs down to one value."""
        if result is None:
            return None

        if isinstance(result, tuple):
            return result[0] if result else None

        if isinstance(result, list):
            if not result:
                return None
            first = result[0]
            if isinstance(first, tuple):
                return first[0] if first else None
            return first

        return result

    def check_agent_name_sql(self, first_name: str, last_name: str, middle_initial: str, title: str):
        """
        Match an agent row while treating empty strings as NULL-equivalent inputs.
        """
        first_name = self._normalize_nullable(first_name)
        last_name = self._normalize_nullable(last_name)
        middle_initial = self._normalize_nullable(middle_initial)
        title = self._normalize_nullable(title)

        sql = """
            SELECT AgentID
            FROM agent
            WHERE
                (FirstName = %s OR (%s IS NULL AND FirstName IS NULL))
                AND (LastName = %s OR (%s IS NULL AND LastName IS NULL))
                AND (MiddleInitial = %s OR (%s IS NULL AND MiddleInitial IS NULL))
                AND (Title = %s OR (%s IS NULL AND Title IS NULL))
        """

        values = [first_name, last_name, middle_initial, title]
        params = tuple(v for value in values for v in (value, value))

        result = self.get_record(sql, params=params)
        return self._extract_scalar(result)

    def check_collector_list(self, collector_list, new_agents=False):
        """
        Ensure collector list is not empty and replace "collector unknown" with unspecified.
        """
        agent_id = self.get_one_match("agent", "AgentID", "LastName", "unspecified")

        unknown_dict = {
            "collector_first_name": "",
            "collector_middle_initial": "",
            "collector_last_name": "unspecified",
            "collector_title": "",
            "agent_id": agent_id,
        }

        if not collector_list and not new_agents:
            collector_list.append(unknown_dict)
        else:
            for index, name_dict in enumerate(collector_list):
                no_agent = any(
                    isinstance(value, str) and value.lower() == "collector unknown"
                    for value in name_dict.values()
                )
                if no_agent:
                    collector_list[index] = unknown_dict if not new_agents else None

            collector_list = [x for x in collector_list if x is not None]

        return collector_list

    def get_collecting_event_ids_by_agent_id(self, agent_id: Union[int, str]):
        """
        Return distinct CollectingEventID values for an AgentID from collector.
        """
        sql = """
            SELECT DISTINCT CollectingEventID
            FROM collector
            WHERE AgentID = %s
              AND CollectingEventID IS NOT NULL
        """
        rows = self.get_records(sql, params=(agent_id,))
        return [row[0] for row in rows if row and row[0] is not None]

    def get_agent_collecting_range(self, first_name: str, last_name: str, middle_initial: str, title: str):
        """
        Find an agent via name fields, get their collecting event ids, then return:
        (max_year, min_year, median_year)
        """
        agent_id = self.check_agent_name_sql(
            first_name=first_name,
            last_name=last_name,
            middle_initial=middle_initial,
            title=title,
        )

        if agent_id is None:
            return None, None, None

        collecting_event_ids = self.get_collecting_event_ids_by_agent_id(agent_id)

        if not collecting_event_ids:
            return None, None, None

        placeholders = ", ".join(["%s"] * len(collecting_event_ids))
        sql = f"""
            SELECT YEAR(StartDate) AS StartDateYear
            FROM collectingevent
            WHERE CollectingEventID IN ({placeholders})
              AND StartDate IS NOT NULL
              AND YEAR(StartDate) IS NOT NULL
        """

        rows = self.get_records(sql, params=tuple(collecting_event_ids))
        years = sorted(int(row[0]) for row in rows if row and row[0] is not None)

        if not years:
            return None, None, None

        med = median(years)
        if isinstance(med, float) and med.is_integer():
            med = int(med)

        return max(years), min(years), med

    def get_one_hybrid(self, match, fullname):
        """
        Match multi-term hybrids irrespective of order.
        """
        parts = match.split()

        if len(parts) == 3:
            basename = fullname.split()[0]

            sql = """
                SELECT TaxonID
                FROM taxon
                WHERE LOWER(FullName) LIKE %s
                  AND LOWER(FullName) LIKE %s
                  AND LOWER(FullName) LIKE %s
                  AND LOWER(FullName) LIKE %s
            """

            params = (
                f"%{parts[0].lower()}%",
                f"%{parts[1].lower()}%",
                f"%{parts[2].lower()}%",
                f"%{basename.lower()}%",
            )

            result = self.get_records(sql=sql, params=params)
            return self._extract_scalar(result)

        elif len(parts) < 3:
            return self.get_one_match(
                tab_name="taxon",
                id_col="TaxonID",
                key_col="FullName",
                match=fullname,
            )

        else:
            self.logger.error("hybrid tax name has more than 3 terms")
            return None

    def get_one_match(self, tab_name, id_col, key_col, match):
        """
        General helper to get one id_col value by exact match on key_col.
        """
        sql = f"SELECT {id_col} FROM {tab_name} WHERE `{key_col}` = %s;"
        result = self.get_record(sql, params=(match,))
        return self._extract_scalar(result)

    def create_insert_statement(self, col_list: list, val_list: list, tab_name: str):
        """
        Create parameterized INSERT statement.
        """
        column_list = ", ".join(col_list)
        placeholders = ", ".join(["%s"] * len(val_list))
        sql = f"INSERT INTO {tab_name} ({column_list}) VALUES ({placeholders});"
        return SqlStatement(sql=sql, params=tuple(val_list) if val_list else None)

    def insert_table_record(self, sql, params=None):
        """
        Execute INSERT/UPDATE style statement.
        """
        cursor = self.get_cursor()
        try:
            if params is not None:
                self.logger.debug(f"running query - {sql} with params {params}")
                cursor.execute(sql, tuple(params))
            else:
                self.logger.debug(f"running query - {sql}")
                cursor.execute(sql)

            self.commit()
        except Exception:
            self.logger.error(traceback.format_exc())
        finally:
            cursor.close()

    def create_batch_record(
        self,
        start_time: datetime,
        end_time: datetime,
        batch_size: int,
        batch_md5: str,
        agent_id: Union[str, int],
    ):
        """
        Create parameterized insert for picturae_batch.
        """
        delt_time = timedelta(seconds=15)
        time_stamp_list = [start_time - delt_time, end_time + delt_time]

        column_list = [
            "batch_MD5",
            "TimestampCreated",
            "TimestampModified",
            "StartTimeStamp",
            "EndTimeStamp",
            "batch_size",
            "CreatedByAgentID",
            "ModifiedByAgentID",
        ]

        value_list = [
            batch_md5,
            time_utils.get_pst_time_now_string(),
            time_utils.get_pst_time_now_string(),
            time_stamp_list[0],
            time_stamp_list[1],
            batch_size,
            agent_id,
            agent_id,
        ]

        value_list, column_list = remove_two_index(value_list, column_list)

        return self.create_insert_statement(
            col_list=column_list,
            val_list=value_list,
            tab_name="picturae_batch",
        )

    def create_update_statement(
        self,
        tab_name,
        agent_id,
        col_list,
        val_list,
        condition_sql,
        condition_params=None,
    ):
        """
        Create parameterized UPDATE statement.

        condition_sql should contain placeholders if needed, e.g.
            "WHERE TaxonID = %s"
        condition_params should be the corresponding values.
        """
        val_list, col_list = remove_two_index(val_list, col_list)

        update_string = " SET TimestampModified = %s, ModifiedByAgentID = %s"
        params = [time_utils.get_pst_time_now_string(), agent_id]

        for col, val in zip(col_list, val_list):
            update_string += f", {col} = %s"
            params.append(val)

        sql = f"UPDATE {tab_name}{update_string} {condition_sql}"

        if condition_params:
            params.extend(condition_params)

        return SqlStatement(sql=sql, params=tuple(params))

    def taxon_get(self, name, hybrid=False, taxname=None):
        """
        Retrieve taxon id from specify database.
        """
        name = name.lower()

        if hybrid is False:
            if "subsp." in name or "var." in name:
                result_id = self.get_one_match(
                    tab_name="taxon",
                    id_col="TaxonID",
                    key_col="FullName",
                    match=name,
                )
                if result_id is None:
                    if "subsp." in name:
                        name = name.replace(" subsp. ", " var. ")
                    elif "var." in name:
                        name = name.replace(" var. ", " subsp. ")

                    result_id = self.get_one_match(
                        tab_name="taxon",
                        id_col="TaxonID",
                        key_col="FullName",
                        match=name,
                    )
            else:
                result_id = self.get_one_match(
                    tab_name="taxon",
                    id_col="TaxonID",
                    key_col="FullName",
                    match=name,
                )

            return result_id

        return self.get_one_hybrid(match=taxname, fullname=name)

    def insert_taxa_added_record(self, taxon_list, df: pd.DataFrame, agent_id: Union[str, int]):
        """
        Insert rows into picturaetaxa_added for newly added taxa.
        """
        taxa_frame = df[df["fullname"].isin(taxon_list)].drop_duplicates(subset=["fullname"])

        for _, row in taxa_frame.iterrows():
            tax_id = self.get_one_match("picturaetaxa_added", "newtaxID", "fullname", row["fullname"])
            if tax_id is None:
                sql_statement = self.create_new_tax_tab(row, "picturaetaxa_added", agent_id)
                self.insert_table_record(sql_statement.sql, sql_statement.params)

    def create_new_tax_tab(self, row, tab_name: str, agent_id: Union[str, int]):
        """
        Create insert statement for new taxon QC table.
        """
        hybrid = string_utils.str_to_bool(row["Hybrid"])

        col_list = [
            "fullname",
            "TimestampCreated",
            "TimestampModified",
            "batch_MD5",
            "family",
            "name",
            "hybrid",
            "CreatedByAgentID",
            "ModifiedByAgentID",
        ]

        val_list = [
            row["fullname"],
            time_utils.get_pst_time_now_string(),
            time_utils.get_pst_time_now_string(),
            row["batch_md5"],
            row["Family"],
            row["taxname"],
            hybrid,
            agent_id,
            agent_id,
        ]

        val_list, col_list = remove_two_index(val_list, col_list)

        return self.create_insert_statement(col_list, val_list, tab_name)

    def get_is_taxon_id_redacted(self, taxon_id):
        """
        Retrieve RedactLocality boolean from vtaxon2.
        """
        sql = "SELECT RedactLocality FROM vtaxon2 WHERE taxonid = %s;"
        result = self.get_record(sql, params=(taxon_id,))

        if result is None:
            self.logger.info(f"taxon id not yet present in vtaxon2: {taxon_id}")
            return False

        val = self._extract_scalar(result)
        return val is True or val == 1 or val == b"\x01"