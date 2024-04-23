import sqlite3

def create_database():
    try:
        # Verbindung zur SQLite-Datenbank herstellen
        conn = sqlite3.connect('C:\\Users\\I551667\\Documents\\GitHub\\StyleSage-Interieur\\backend\\app\\Datenbank\\StyleSage.db')
        cursor = conn.cursor()

        # Tabelle für Raumfotos erstellen
        cursor.execute('''CREATE TABLE IF NOT EXISTS room_photos (
                            id INTEGER PRIMARY KEY,
                            photo_name TEXT,
                            style_id INTEGER,
                            FOREIGN KEY (style_id) REFERENCES interior_styles (id)
                          )''')

        # Tabelle für Möbelfotos erstellen
        cursor.execute('''CREATE TABLE IF NOT EXISTS furniture_photos (
                            id INTEGER PRIMARY KEY,
                            name TEXT,
                            photo_url TEXT,
                            shop_url TEXT,
                            style_id INTEGER,
                            furniture_id INTEGER,
                            FOREIGN KEY (style_id) REFERENCES interior_styles (id)
                            FOREIGN KEY (furniture_id) REFERENCES furniture_styles (id)
                          )''')

        # Tabelle für Einrichtungsgegenstände erstellen
        cursor.execute('''CREATE TABLE IF NOT EXISTS furniture_styles (
                            id INTEGER PRIMARY KEY,
                            style_name TEXT
                          )''')

        # Tabelle für Einrichtungsstile erstellen
        cursor.execute('''CREATE TABLE IF NOT EXISTS interior_styles (
                            id INTEGER PRIMARY KEY,
                            style_name TEXT
                          )''')
        
        cursor.execute('''CREATE TABLE IF NOT EXISTS sofa_collection (
                            id INTEGER PRIMARY KEY,
                            photo_name TEXT,
                            style_id INTEGER,
                            FOREIGN KEY (style_id) REFERENCES interior_styles (id)
                          )''')
        
        cursor.execute('''CREATE TABLE IF NOT EXISTS schrank_collection (
                            id INTEGER PRIMARY KEY,
                            photo_name TEXT,
                            style_id INTEGER,
                            FOREIGN KEY (style_id) REFERENCES interior_styles (id)
                          )''')
        
        cursor.execute('''CREATE TABLE IF NOT EXISTS bett_collection (
                            id INTEGER PRIMARY KEY,
                            photo_name TEXT,
                            style_id INTEGER,
                            FOREIGN KEY (style_id) REFERENCES interior_styles (id)
                          )''')
        
        cursor.execute('''CREATE TABLE IF NOT EXISTS deko_collection (
                            id INTEGER PRIMARY KEY,
                            photo_name TEXT,
                            style_id INTEGER,
                            FOREIGN KEY (style_id) REFERENCES interior_styles (id)
                          )''')
        
        cursor.execute('''CREATE TABLE IF NOT EXISTS beleuchtung_collection (
                            id INTEGER PRIMARY KEY,
                            photo_name TEXT,
                            style_id INTEGER,
                            FOREIGN KEY (style_id) REFERENCES interior_styles (id)
                          )''')
        
        cursor.execute('''CREATE TABLE IF NOT EXISTS tisch_collection (
                            id INTEGER PRIMARY KEY,
                            photo_name TEXT,
                            style_id INTEGER,
                            FOREIGN KEY (style_id) REFERENCES interior_styles (id)
                          )''')
        
        cursor.execute('''CREATE TABLE IF NOT EXISTS stuhl_collection (
                            id INTEGER PRIMARY KEY,
                            photo_name TEXT,
                            style_id INTEGER,
                            FOREIGN KEY (style_id) REFERENCES interior_styles (id)
                          )''')

        # Änderungen bestätigen und Verbindung schließen
        conn.commit()
        conn.close()
        print("Datenbank erfolgreich erstellt.")
    except sqlite3.Error as e:
        print("Fehler beim Erstellen der Datenbank:", e)
