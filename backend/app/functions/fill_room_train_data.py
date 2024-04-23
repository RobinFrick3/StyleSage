import sqlite3

def fill_room_train_database():
    try:
        # Verbindung zur SQLite-Datenbank herstellen
        conn = sqlite3.connect('C:\\Users\\I551667\\Documents\\GitHub\\StyleSage-Interieur\\backend\\app\\Datenbank\\StyleSage.db')
        cursor = conn.cursor()

        # Daten für Raumfotos einfügen
        for i in range(1, 1772):
            cursor.execute("INSERT INTO room_photos (photo_name, style_id) VALUES (?, ?)", (f'modern_{i}.jpg', 1))
        for i in range(1, 2175):
            cursor.execute("INSERT INTO room_photos (photo_name, style_id) VALUES (?, ?)", (f'maritim_{i}.jpg', 2))
        for i in range(1, 1304):
            cursor.execute("INSERT INTO room_photos (photo_name, style_id) VALUES (?, ?)", (f'minimalistisch_{i}.jpg', 3))
        for i in range(1, 1748):
            cursor.execute("INSERT INTO room_photos (photo_name, style_id) VALUES (?, ?)", (f'industrial_{i}.jpg', 4))
        for i in range(1, 1775):
            cursor.execute("INSERT INTO room_photos (photo_name, style_id) VALUES (?, ?)", (f'rustikal_{i}.jpg', 5))
        for i in range(1, 1201):
            cursor.execute("INSERT INTO room_photos (photo_name, style_id) VALUES (?, ?)", (f'boho_{i}.jpg', 6))

        # Änderungen bestätigen und Verbindung schließen
        conn.commit()
        conn.close()
        print("Datenbank erfolgreich befüllt.")
    except sqlite3.Error as e:
        print("Fehler beim Befüllen der Datenbank:", e)
