#!/usr/bin/env python3
import os
import sqlite3
from sqlite3 import Error

DB_PATH = os.path.join(os.path.dirname(__file__), 'db//park.db')


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


def insert_lot(conn, name, latitude, longitude, active):
    query = """ INSERT INTO Lot(Name, Latitude, Longitude, Active) VALUES(?, ?, ?, ?) """
    data = (name, latitude, longitude, active)
    cur = conn.cursor()
    cur.execute(query, data)
    return cur.lastrowid


def insert_source(conn, URI, username, password, location, active):
    query = """ INSERT INTO Source(URI, Username, Password, Location, Active) VALUES(?, ?, ?, ?, ?) """
    data = (URI, username, password, location, active)
    cur = conn.cursor()
    cur.execute(query, data)
    return cur.lastrowid


def insert_type(conn, description):
    query = ''' INSERT INTO Type(Description) VALUES(?) '''
    data = (description)
    cur = conn.cursor()
    # Must use braces when passing only one value, or it will think the string is a char array
    cur.execute(query, [description])
    return cur.lastrowid


def insert_zone(conn, lotID, sourceID, typeID, totalSpaces, polyCoords, active):
    query = """ INSERT INTO Zone(LotID, SourceID, TypeID, TotalSpaces, PolyCoords, Active) VALUES(?, ?, ?, ?, ?, ?) """
    data = (lotID, sourceID, typeID, totalSpaces, polyCoords, active)
    cur = conn.cursor()
    cur.execute(query, data)
    return cur.lastrowid


def insert_occupancy_log(conn, timestamp, zoneID, typeID, lotID, occupiedSpaces, totalSpaces):
    query = """ INSERT INTO OccupancyLog(Timestamp, ZoneID, TypeID, LotID, OccupiedSpaces, TotalSpaces) VALUES(?, ?, ?, ?, ?, ?) """
    data = (timestamp, zoneID, typeID, lotID, occupiedSpaces, totalSpaces)
    cur = conn.cursor()
    cur.execute(query, data)
    return cur.lastrowid


def main():
    # Connect to the database
    print(DB_PATH)
    conn = db_connect(DB_PATH)
    if conn is not None:
        with conn:
            # Create tables
            create_db_table(conn, """ CREATE TABLE IF NOT EXISTS Lot (
                                        LotID integer PRIMARY KEY,
                                        Name text NOT NULL UNIQUE,
                                        Latitude real NOT NULL DEFAULT '0.0',
                                        Longitude real NOT NULL DEFAULT '0.0',
                                        Active integer NOT NULL  DEFAULT '0' CHECK (Active >= 0 OR Active <= 1)
                                    );""")
            create_db_table(conn, """ CREATE TABLE IF NOT EXISTS Source (
                                        SourceID integer PRIMARY KEY,
                                        URI text NOT NULL UNIQUE,
                                        Username text,
                                        Password text,
                                        Location text NOT NULL,
                                        Active integer NOT NULL DEFAULT '0' CHECK (Active >= 0 OR Active <= 1)
                                    );""")
            create_db_table(conn, """ CREATE TABLE IF NOT EXISTS Type (
                                        TypeID integer PRIMARY KEY,
                                        Description text NOT NULL UNIQUE
                                    ); """)
            create_db_table(conn, """ CREATE TABLE IF NOT EXISTS Zone (
                                        ZoneID integer PRIMARY KEY,
                                        LotID integer NOT NULL,
                                        SourceID integer NOT NULL,
                                        TypeID integer NOT NULL,
                                        TotalSpaces integer NOT NULL DEFAULT '0' CHECK (TotalSpaces >= 0),
                                        PolyCoords text NOT NULL UNIQUE,
                                        Active integer NOT NULL DEFAULT '0' CHECK (Active >= 0 OR Active <= 1),
                                        FOREIGN KEY (LotID) REFERENCES Lot (LotID),
                                        FOREIGN KEY (SourceID) REFERENCES Source (SourceID),
                                        FOREIGN KEY (TypeID) REFERENCES Type (TypeID),
                                        UNIQUE (ZoneID, LotID, SourceID, TypeID)
                                    );""")
            create_db_table(conn, """ CREATE TABLE IF NOT EXISTS OccupancyLog (
                                        Timestamp real NOT NULL,
                                        ZoneID integer NOT NULL,
                                        TypeID integer NOT NULL,
                                        LotID integer NOT NULL,
                                        OccupiedSpaces integer NOT NULL DEFAULT '0' CHECK (OccupiedSpaces >= 0 AND OccupiedSpaces <= TotalSpaces),
                                        TotalSpaces integer NOT NULL DEFAULT '0' CHECK (TotalSpaces >= 0 AND TotalSpaces >= OccupiedSpaces),
                                        FOREIGN KEY (ZoneID) REFERENCES Zone (ZoneID),
                                        FOREIGN KEY (TypeID) REFERENCES Type (TypeID),
                                        FOREIGN KEY (LotID) REFERENCES Lot (LotID),
                                        UNIQUE (Timestamp, ZoneID, TypeID, LotID)
                                    );""")
            # Insert initial values
            insert_lot(conn, 'Lot01', 38.364554, -75.601320, 1)
            insert_source(conn, 'https://raw.githubusercontent.com/garciart/Park/master/demos/demo_images/demo_imagex1.jpg',
                          '', '', 'Salisbury Parking Garage West', 1)
            insert_type(conn, 'General')
            insert_type(conn, 'Handicap')
            insert_type(conn, 'Employee')
            insert_type(conn, 'Visitor')
            insert_zone(
                conn, 1, 1, 3, 9, '[[816, 1150], [3200, 1140], [3200, 1350], [816, 1400]]', 1)
            insert_zone(
                conn, 1, 1, 2, 2, '[[240, 1140], [815, 1150], [815, 1400], [150, 1400]]', 1)
    else:
        print("Error! Cannot connect to the database!")


if __name__ == '__main__':
    main()
