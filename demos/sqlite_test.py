#!python
import os
import sqlite3
from sqlite3 import Error

DB_PATH = os.path.join(os.path.dirname(__file__), 'db\\test.db')


def db_connect(db_file=DB_PATH):
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)

    return conn


def create_db_table(conn, query):
    try:
        c = conn.cursor()
        c.execute(query)
    except Error as e:
        print(e)


def main():
    # Connect to the database
    conn = db_connect(DB_PATH)
    if conn is not None:
        create_db_table(conn, """ CREATE TABLE IF NOT EXISTS Lot (
                                    LotID integer PRIMARY KEY,
                                    Name text NOT NULL,
                                    Latitude real NOT NULL,
                                    Longitude real NOT NULL
                                );""")
        create_db_table(conn, """ CREATE TABLE IF NOT EXISTS Source (
                                    SourceID integer PRIMARY KEY,
                                    URI text NOT NULL,
                                    Username text NOT NULL,
                                    Password text NOT NULL,
                                    Active integer NOT NULL,
                                    Location text NOT NULL
                                );""")
        create_db_table(conn, """ CREATE TABLE IF NOT EXISTS Type (
                                    TypeID integer PRIMARY KEY,
                                    Description text NOT NULL
                                    ); """)
        create_db_table(conn, """ CREATE TABLE IF NOT EXISTS Zone (
                                    ZoneID integer PRIMARY KEY,
                                    LotID integer NOT NULL,
                                    SourceID integer NOT NULL,
                                    TypeID integer NOT NULL,
                                    MaxSpaces integer NOT NULL,
                                    PolyCoords text NOT NULL
                                );""")
    else:
        print("Error! Cannot connect to the database!")


if __name__ == '__main__':
    main()
