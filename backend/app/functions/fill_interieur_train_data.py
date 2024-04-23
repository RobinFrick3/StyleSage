import sqlite3

def fill_interieur_train_database():
    try:
        # Verbindung zur SQLite-Datenbank herstellen
        conn = sqlite3.connect('C:\\Users\\I551667\\Documents\\GitHub\\StyleSage-Interieur\\backend\\app\\Datenbank\\StyleSage.db')
        cursor = conn.cursor()

        # Daten für Sofafotos einfügen
        for i in range(1, 2):
            cursor.execute("INSERT INTO sofa_collection (photo_name, style_id) VALUES (?, ?)", (f'modern_{i}.jpg', 1))
        for i in range(1, 2):
            cursor.execute("INSERT INTO sofa_collection (photo_name, style_id) VALUES (?, ?)", (f'maritim_{i}.jpg', 2))
        for i in range(1, 2):
            cursor.execute("INSERT INTO sofa_collection (photo_name, style_id) VALUES (?, ?)", (f'minimalistisch_{i}.jpg', 3))
        for i in range(1, 2):
            cursor.execute("INSERT INTO sofa_collection (photo_name, style_id) VALUES (?, ?)", (f'industrial_{i}.jpg', 4))
        for i in range(1, 2):
            cursor.execute("INSERT INTO sofa_collection (photo_name, style_id) VALUES (?, ?)", (f'rustikal_{i}.jpg', 5))
        for i in range(1, 2):
            cursor.execute("INSERT INTO sofa_collection (photo_name, style_id) VALUES (?, ?)", (f'boho_{i}.jpg', 6))

        # Daten für Schrankfotos einfügen
        for i in range(1, 142):
            cursor.execute("INSERT INTO schrank_collection (photo_name, style_id) VALUES (?, ?)", (f'modern_{i}.jpg', 1))
        for i in range(1, 14):
            cursor.execute("INSERT INTO schrank_collection (photo_name, style_id) VALUES (?, ?)", (f'maritim_{i}.jpg', 2))
        for i in range(1, 26):
            cursor.execute("INSERT INTO schrank_collection (photo_name, style_id) VALUES (?, ?)", (f'minimalistisch_{i}.jpg', 3))
        for i in range(1, 140):
            cursor.execute("INSERT INTO schrank_collection (photo_name, style_id) VALUES (?, ?)", (f'industrial_{i}.jpg', 4))
        for i in range(1, 71):
            cursor.execute("INSERT INTO schrank_collection (photo_name, style_id) VALUES (?, ?)", (f'rustikal_{i}.jpg', 5))
        for i in range(1, 87):
            cursor.execute("INSERT INTO schrank_collection (photo_name, style_id) VALUES (?, ?)", (f'boho_{i}.jpg', 6))

        # Daten für Bettfotos einfügen
        for i in range(1, 81):
            cursor.execute("INSERT INTO bett_collection (photo_name, style_id) VALUES (?, ?)", (f'modern_{i}.jpg', 1))
        for i in range(1, 35):
            cursor.execute("INSERT INTO bett_collection (photo_name, style_id) VALUES (?, ?)", (f'maritim_{i}.jpg', 2))
        for i in range(1, 27):
            cursor.execute("INSERT INTO bett_collection (photo_name, style_id) VALUES (?, ?)", (f'minimalistisch_{i}.jpg', 3))
        for i in range(1, 45):
            cursor.execute("INSERT INTO bett_collection (photo_name, style_id) VALUES (?, ?)", (f'industrial_{i}.jpg', 4))
        for i in range(1, 20):
            cursor.execute("INSERT INTO bett_collection (photo_name, style_id) VALUES (?, ?)", (f'rustikal_{i}.jpg', 5))
        for i in range(1, 43):
            cursor.execute("INSERT INTO bett_collection (photo_name, style_id) VALUES (?, ?)", (f'boho_{i}.jpg', 6))

        # Daten für Dekofotos einfügen
        for i in range(1, 54):
            cursor.execute("INSERT INTO deko_collection (photo_name, style_id) VALUES (?, ?)", (f'modern_{i}.jpg', 1))
        for i in range(1, 13):
            cursor.execute("INSERT INTO deko_collection (photo_name, style_id) VALUES (?, ?)", (f'maritim_{i}.jpg', 2))
        for i in range(1, 25):
            cursor.execute("INSERT INTO deko_collection (photo_name, style_id) VALUES (?, ?)", (f'minimalistisch_{i}.jpg', 3))
        for i in range(1, 54):
            cursor.execute("INSERT INTO deko_collection (photo_name, style_id) VALUES (?, ?)", (f'industrial_{i}.jpg', 4))
        for i in range(1, 23):
            cursor.execute("INSERT INTO deko_collection (photo_name, style_id) VALUES (?, ?)", (f'rustikal_{i}.jpg', 5))
        for i in range(1, 70):
            cursor.execute("INSERT INTO deko_collection (photo_name, style_id) VALUES (?, ?)", (f'boho_{i}.jpg', 6))

        # Daten für Beleuchtungfotos einfügen
        for i in range(1, 20):
            cursor.execute("INSERT INTO beleuchtung_collection (photo_name, style_id) VALUES (?, ?)", (f'modern_{i}.jpg', 1))
        for i in range(1, 8):
            cursor.execute("INSERT INTO beleuchtung_collection (photo_name, style_id) VALUES (?, ?)", (f'maritim_{i}.jpg', 2))
        for i in range(1, 15):
            cursor.execute("INSERT INTO beleuchtung_collection (photo_name, style_id) VALUES (?, ?)", (f'minimalistisch_{i}.jpg', 3))
        for i in range(1, 68):
            cursor.execute("INSERT INTO beleuchtung_collection (photo_name, style_id) VALUES (?, ?)", (f'industrial_{i}.jpg', 4))
        for i in range(1, 15):
            cursor.execute("INSERT INTO beleuchtung_collection (photo_name, style_id) VALUES (?, ?)", (f'rustikal_{i}.jpg', 5))
        for i in range(1, 40):
            cursor.execute("INSERT INTO beleuchtung_collection (photo_name, style_id) VALUES (?, ?)", (f'boho_{i}.jpg', 6))

        # Daten für Tischfotos einfügen
        for i in range(1, 153):
            cursor.execute("INSERT INTO tisch_collection (photo_name, style_id) VALUES (?, ?)", (f'modern_{i}.jpg', 1))
        for i in range(1, 36):
            cursor.execute("INSERT INTO tisch_collection (photo_name, style_id) VALUES (?, ?)", (f'maritim_{i}.jpg', 2))
        for i in range(1, 20):
            cursor.execute("INSERT INTO tisch_collection (photo_name, style_id) VALUES (?, ?)", (f'minimalistisch_{i}.jpg', 3))
        for i in range(1, 148):
            cursor.execute("INSERT INTO tisch_collection (photo_name, style_id) VALUES (?, ?)", (f'industrial_{i}.jpg', 4))
        for i in range(1, 153):
            cursor.execute("INSERT INTO tisch_collection (photo_name, style_id) VALUES (?, ?)", (f'rustikal_{i}.jpg', 5))
        for i in range(1, 35):
            cursor.execute("INSERT INTO tisch_collection (photo_name, style_id) VALUES (?, ?)", (f'boho_{i}.jpg', 6))

        # Daten für Stuhlfotos einfügen
        for i in range(1, 145):
            cursor.execute("INSERT INTO stuhl_collection (photo_name, style_id) VALUES (?, ?)", (f'modern_{i}.jpg', 1))
        for i in range(1, 18):
            cursor.execute("INSERT INTO stuhl_collection (photo_name, style_id) VALUES (?, ?)", (f'maritim_{i}.jpg', 2))
        for i in range(1, 18):
            cursor.execute("INSERT INTO stuhl_collection (photo_name, style_id) VALUES (?, ?)", (f'minimalistisch_{i}.jpg', 3))
        for i in range(1, 202):
            cursor.execute("INSERT INTO stuhl_collection (photo_name, style_id) VALUES (?, ?)", (f'industrial_{i}.jpg', 4))
        for i in range(1, 132):
            cursor.execute("INSERT INTO stuhl_collection (photo_name, style_id) VALUES (?, ?)", (f'rustikal_{i}.jpg', 5))
        for i in range(1, 37):
            cursor.execute("INSERT INTO stuhl_collection (photo_name, style_id) VALUES (?, ?)", (f'boho_{i}.jpg', 6))

        # Änderungen bestätigen und Verbindung schließen
        conn.commit()
        conn.close()
        print("Datenbank erfolgreich befüllt.")
    except sqlite3.Error as e:
        print("Fehler beim Befüllen der Datenbank:", e)
