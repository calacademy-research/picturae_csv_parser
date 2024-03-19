#!/bin/bash
touch casbotany_lite.db
# creating tables
python casbotany_lite_creator.py
# gets data dump of taxon tree
mysqldump -u [username] -p -h [hostname] [database] taxon > taxon_dump.sql
# will need to insert steps to clean up taxon_dump.sql
python convert_mysql_sqlite.py -f taxon_dump.sql
# creating taxon tree
sqlite3 casbotany_lite.db < taxon_dump.sql
# sometime it takes a couple seconds for the sqlite db to appear


