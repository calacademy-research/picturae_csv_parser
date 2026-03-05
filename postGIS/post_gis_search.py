import re
import unicodedata
from difflib import SequenceMatcher

import psycopg2
import pandas as pd


class GadmLookup:
    def __init__(
        self,
        host="localhost",
        dbname="gis",
        user="postgres",
        password="postgres",
        port=5432,
        adm1_table="public.gadm",
        country_aliases=None,
        state_aliases=None,
    ):
        self.adm1_table = adm1_table
        self.conn = psycopg2.connect(
            host=host,
            dbname=dbname,
            user=user,
            password=password,
            port=port,
        )

        # Canonicalize *declared* values (your DB side).
        self.country_aliases = country_aliases or {
            "United States": "United States of America",
            "USA": "United States of America",
            "U.S.A.": "United States of America",
            "US": "United States of America",
            "México": "Mexico",
            "Mexico": "Mexico",
        }

        self.state_aliases = state_aliases or {
            # add as needed
        }

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass

    def lookup_admin_div(self, lat, lon):
        if lat is None or lon is None:
            return None
        if pd.isna(lat) or pd.isna(lon):
            return None

        with self.conn.cursor() as cur:
            sql = f"""
            SELECT
                "name_0" AS country_name,
                "name_1" AS admin1_name,
                "gid_0"  AS country_gid,
                "gid_1"  AS admin1_gid
            FROM {self.adm1_table}
            WHERE ST_Intersects(
                geom,
                ST_SetSRID(ST_MakePoint(%s, %s), 4326)
            )
            LIMIT 1;
            """
            cur.execute(sql, (float(lon), float(lat)))
            row = cur.fetchone()

        if not row:
            return None

        return {
            "gadm_country": row[0],
            "gadm_admin1": row[1],
            "gadm_gid_0": row[2],
            "gadm_gid_1": row[3],
        }

    @staticmethod
    def _strip_diacritics(s: str) -> str:
        # México -> Mexico, São -> Sao, etc.
        s = unicodedata.normalize("NFKD", s)
        return "".join(ch for ch in s if not unicodedata.combining(ch))

    @classmethod
    def normalize_name(cls, value):
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return ""

        s = str(value).strip()
        if not s:
            return ""

        s = s.casefold()
        s = cls._strip_diacritics(s)  # <-- key change
        s = re.sub(r"[^\w\s]", " ", s)

        replacements = {
            "province of ": "",
            "state of ": "",
            "departamento de ": "",
            "departamento del ": "",
            "county of ": "",
            "region of ": "",
            "provincia de ": "",
            "provincia del ": "",
            " municipality": "",
            " prefecture": "",
        }

        for old, new in replacements.items():
            s = s.replace(old, new)

        s = re.sub(r"\s+", " ", s).strip()
        return s

    @classmethod
    def fuzzy_match(cls, a, b, threshold=0.88):
        na = cls.normalize_name(a)
        nb = cls.normalize_name(b)

        if not na or not nb:
            return False
        if na == nb:
            return True
        if na in nb or nb in na:
            return True

        score = SequenceMatcher(None, na, nb).ratio()
        return score >= threshold

    def canonical_country(self, value):
        if value is None:
            return ""
        raw = str(value).strip()
        return self.country_aliases.get(raw, raw)

    def canonical_state(self, value):
        if value is None:
            return ""
        raw = str(value).strip()
        return self.state_aliases.get(raw, raw)

    def validate_country_admin1(self, declared_country, declared_state, gadm_result):
        if not gadm_result:
            return False

        gadm_country_raw = gadm_result.get("gadm_country", "")
        gadm_admin1_raw = gadm_result.get("gadm_admin1", "")

        declared_country_cmp = self.canonical_country(declared_country)
        declared_state_cmp = self.canonical_state(declared_state)

        # Optional: canonicalize GADM side too (helps if you add more aliases later)
        gadm_country_cmp = self.canonical_country(gadm_country_raw)
        gadm_admin1_cmp = self.canonical_state(gadm_admin1_raw)

        # --- Special-case: Taiwan recorded as a "state" under China in your DB ---
        decl_country_norm = self.normalize_name(declared_country_cmp)
        decl_state_norm = self.normalize_name(declared_state_cmp)
        gadm_country_norm = self.normalize_name(gadm_country_cmp)

        if (
            decl_country_norm in {"china", "people s republic of china", "prc"}
            and decl_state_norm in {"taiwan", "taiwan province", "province of taiwan"}
            and gadm_country_norm == "taiwan"
        ):
            # coords are in Taiwan; accept without forcing NAME_1 match
            return True

        country_ok = self.fuzzy_match(declared_country_cmp, gadm_country_cmp, threshold=0.90)

        state_has_value = self.normalize_name(declared_state_cmp) != ""
        if state_has_value:
            state_ok = self.fuzzy_match(declared_state_cmp, gadm_admin1_cmp, threshold=0.85)
        else:
            state_ok = True

        return bool(country_ok and state_ok)