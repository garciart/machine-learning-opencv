DROP TABLE IF EXISTS Lot;
DROP TABLE IF EXISTS Source;
DROP TABLE IF EXISTS Type;
DROP TABLE IF EXISTS Zone;
DROP TABLE IF EXISTS OccupancyLog;
CREATE TABLE IF NOT EXISTS Lot (
    LotID INTEGER PRIMARY KEY,
    Name TEXT NOT NULL UNIQUE,
    Latitude REAL NOT NULL DEFAULT '0.0',
    Longitude REAL NOT NULL DEFAULT '0.0',
    Active INTEGER NOT NULL DEFAULT '0'
);
CREATE TABLE IF NOT EXISTS Source (
    SourceID INTEGER PRIMARY KEY,
    URI TEXT NOT NULL UNIQUE,
    Username TEXT NOT NULL,
    Password TEXT NOT NULL,
    Location TEXT NOT NULL,
    Active INTEGER NOT NULL DEFAULT '0'
);
CREATE TABLE IF NOT EXISTS Type (
    TypeID INTEGER PRIMARY KEY,
    Description TEXT NOT NULL UNIQUE
);
CREATE TABLE IF NOT EXISTS Zone (
    ZoneID INTEGER PRIMARY KEY,
    LotID INTEGER NOT NULL,
    SourceID INTEGER NOT NULL,
    TypeID INTEGER NOT NULL,
    TotalSpaces INTEGER NOT NULL DEFAULT '0',
    PolyCoords TEXT NOT NULL UNIQUE,
    Active INTEGER NOT NULL DEFAULT '0',
    FOREIGN KEY (LotID) REFERENCES Lot (LotID),
    FOREIGN KEY (SourceID) REFERENCES Source (SourceID),
    FOREIGN KEY (TypeID) REFERENCES Type (TypeID),
	UNIQUE (ZoneID, LotID, SourceID, TypeID)
);
CREATE TABLE IF NOT EXISTS OccupancyLog (
    Timestamp REAL NOT NULL,
    ZoneID INTEGER NOT NULL,
    TypeID INTEGER NOT NULL,
    LotID INTEGER NOT NULL,
    OccupiedSpaces INTEGER NOT NULL DEFAULT '0',
    TotalSpaces INTEGER NOT NULL DEFAULT '0',
    FOREIGN KEY (ZoneID) REFERENCES Zone (ZoneID),
    FOREIGN KEY (TypeID) REFERENCES Type (TypeID),
    FOREIGN KEY (LotID) REFERENCES Lot (LotID),
	UNIQUE (Timestamp, ZoneID, TypeID, LotID)
);
INSERT INTO Lot(Name, Latitude, Longitude, Active) VALUES('Lot01', 38.364554, -75.601320, 1);
INSERT INTO Source(URI, Username, Password, Location, Active) VALUES('https://raw.githubusercontent.com/garciart/Park/master/demos/demo_images/demo_image.jpg', '', '', 'Salisbury Parking Garage West', 1);
INSERT INTO Type(Description) VALUES('General');
INSERT INTO Type(Description) VALUES('Handicap');
INSERT INTO Type(Description) VALUES('Employee');
INSERT INTO Type(Description) VALUES('Visitor');
INSERT INTO Zone(LotID, SourceID, TypeID, TotalSpaces, PolyCoords, Active) VALUES(1, 1, 3, 9, '[[751, 1150], [3200, 1140], [3200, 1350], [816, 1400], [816, 1300]]', 1);
INSERT INTO Zone(LotID, SourceID, TypeID, TotalSpaces, PolyCoords, Active) VALUES(1, 1, 2, 2, '[[150, 1400], [815, 1400], [815, 1300], [750, 1150], [240, 1140]]', 1);

