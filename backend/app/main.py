from fastapi import FastAPI, Request, File, UploadFile, Query  # Importieren der benötigten FastAPI-Module für die Erstellung einer Web-API
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse, JSONResponse, Response  # Importieren der verschiedenen Response-Typen für die API
from fastapi.staticfiles import StaticFiles  # Importieren der StaticFiles-Klasse zum Bereitstellen statischer Dateien
from fastapi.templating import Jinja2Templates  # Importieren der Jinja2Templates-Klasse zum Rendern von HTML-Templates
from .brain.classifier_coach import train_model, fetch_data_from_database  # Importieren von Funktionen aus dem 'brain'-Modul
from .brain.sondi_classifier_coach import train_sondi_model  # Importieren der Funktion zum Trainieren des Sondi-Modells
from .brain.classifier import classify_image_from_file  # Importieren der Funktion zum Klassifizieren von Bildern
from .functions.Segmentierer import segment_image  # Importieren der Funktion zum Segmentieren von Bildern
from .functions.sondi_calculator import calculate_Style  # Importieren der Funktion zum Berechnen von Stilen
from .functions.database import create_database  # Importieren der Funktion zum Erstellen der Datenbank
from .functions.fill_show_data import fill_show_database  # Importieren der Funktion zum Befüllen der Show-Datenbank
from .functions.fill_room_train_data import fill_room_train_database  # Importieren der Funktion zum Befüllen der Raum-Trainingsdatenbank
from .functions.fill_interieur_train_data import fill_interieur_train_database  # Importieren der Funktion zum Befüllen der Interieur-Trainingsdatenbank
import os  # Importieren des 'os'-Moduls für Betriebssystemoperationen
import tempfile  # Importieren des 'tempfile'-Moduls zum Arbeiten mit temporären Dateien
import sqlite3  # Importieren des 'sqlite3'-Moduls zum Arbeiten mit SQLite-Datenbanken
import json  # Importieren des 'json'-Moduls zum Arbeiten mit JSON-Daten

app = FastAPI()  # Erstellen einer FastAPI-Instanz für die Webanwendung

create_database()  # Aufrufen der Funktion zum Erstellen der Datenbank
#fill_interieur_train_database()
#fill_room_train_database()
#fill_show_database()
doTraining = False  # Festlegen eines Flags für das Trainieren des Modells
doSondiTraining = False  # Festlegen eines Flags für das Trainieren des Sondi-Modells

# Bestimmen des Verzeichnisses dieser Datei und der Verzeichnisse für das Frontend und die statischen Dateien
current_directory = os.path.dirname(os.path.abspath(__file__))
frontend_directory = os.path.join(current_directory, "../../frontend")
static_directory = os.path.join(frontend_directory, "static")

# Bereitstellen des statischen Verzeichnisses für die Webanwendung
app.mount("/static", StaticFiles(directory=static_directory), name="static")

# Initialisieren der Vorlagendateien für die Webseiten
templates = Jinja2Templates(directory=os.path.join(frontend_directory, "templates"))

model_path = 'trained_model.pth'  # Pfad zum trainierten Modell
temp_file_path = None  # Initialisieren des temporären Dateipfads für Bilder

# Funktion, die beim Starten der Webanwendung aufgerufen wird
@app.on_event("startup")
async def startup_event():
    # Trainieren des Modells, falls erforderlich
    if doTraining:
        global model_path
        data = fetch_data_from_database()
        train_model(data, model_path)

    # Trainieren des Sondi-Modells, falls erforderlich
    if doSondiTraining:
        train_sondi_model('sofa_collection')
        train_sondi_model('tisch_collection')
        train_sondi_model('schrank_collection')
        train_sondi_model('bett_collection')
        train_sondi_model('stuhl_collection')
        train_sondi_model('beleuchtung_collection')
        train_sondi_model('deko_collection')
    
# Endpunkt für die Startseite der Webanwendung
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Endpunkt für die Upload-Seite der Webanwendung
@app.get("/Upload", response_class=HTMLResponse)
async def read_upload(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

# Endpunkt für das Hochladen von Bildern
@app.post("/Upload")
async def upload_image(request: Request, imageFile: UploadFile = File(...)):
    global temp_file_path
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False)  # Erstellen einer temporären Datei
        temp_file_name = temp_file.name

        # Schreiben der Bilddaten in die temporäre Datei
        with open(temp_file_name, "wb") as buffer:
            buffer.write(await imageFile.read())

        temp_file_path = temp_file_name  # Setzen des temporären Dateipfads

        return RedirectResponse(url="/Analyse")  # Weiterleitung zur Analyse-Seite

    except Exception as e:
        return {"error": str(e)}  # Fehlermeldung zurückgeben, falls ein Fehler auftritt

# Endpunkt für die Analyse-Seite der Webanwendung
@app.get("/Analyse", response_class=HTMLResponse)
async def read_next_page(request: Request):
    global temp_file_path
    if temp_file_path:
        return templates.TemplateResponse("analyse.html", {"request": request, "temp_file_path": temp_file_path})
    else:
        return {"message": "Temporäres Bild nicht gefunden."}  # Meldung zurückgeben, wenn kein temporäres Bild gefunden wurde

# Endpunkt zum Abrufen des temporären Bildes
@app.get("/temp_image")
async def get_temp_image():
    global temp_file_path
    if temp_file_path:
        return FileResponse(temp_file_path)
    else:
        return {"message": "Temporäres Bild nicht gefunden."}  # Meldung zurückgeben, wenn kein temporäres Bild gefunden wurde

# Endpunkt zur Kategorisierung des Bildstils
@app.get("/style_category")
async def get_style_category(request: Request) -> JSONResponse:
    try:
        if temp_file_path is None:
            raise ValueError("No image file path provided.")  # Fehlermeldung, falls kein Bildpfad vorhanden ist
        
        predicted_label = classify_image_from_file(temp_file_path)  # Bildklassifizierung durchführen
        predicted_sondi_label = calculate_Style(segment_image(temp_file_path)) # Zweite Bildklassifizierung durchführen
        print(predicted_label)
        print(predicted_sondi_label)

        if predicted_sondi_label == None:
            return None

        conn = sqlite3.connect('C:\\Users\\I551667\\Documents\\GitHub\\StyleSage-Interieur\\backend\\app\\Datenbank\\StyleSage.db')  # Verbindung zur SQLite-Datenbank herstellen
        cursor = conn.cursor()

        cursor.execute("SELECT id, style_name FROM interior_styles WHERE id = ?", (predicted_label,))  # Abfrage zur Stylekategorie ausführen
        data = cursor.fetchone()

        conn.close()  # Verbindung schließen

        return JSONResponse(content={"style_id": data[0], "style_name": data[1]})  # Daten zurückgeben
    except Exception as e:
        print(e)  # Fehlermeldung ausgeben
        return JSONResponse(content={"error": str(e)}, status_code=500)  # Fehlermeldung zurückgeben

# Endpunkt zum Abrufen von Daten aus der Datenbank
@app.get("/get_data_from_database")
async def get_data_from_database(furniture_type: int = Query(None), style_type: int = Query(None)):
    try:
        conn = sqlite3.connect('C:\\Users\\I551667\\Documents\\GitHub\\StyleSage-Interieur\\backend\\app\\Datenbank\\StyleSage.db')  # Verbindung zur SQLite-Datenbank herstellen
        cursor = conn.cursor()

        if furniture_type is None:
            cursor.execute("SELECT f.photo_url, f.shop_url, f.name \
                            FROM furniture_photos AS f \
                            WHERE style_id = ?", (style_type,))  # Abfrage ohne Möbelkategorie ausführen
        else:
            cursor.execute("SELECT f.photo_url, f.shop_url, f.name \
                            FROM furniture_photos AS f \
                            WHERE  furniture_id = ? AND style_id = ?", (furniture_type, style_type,))  # Abfrage mit Möbelkategorie ausführen
        
        data = cursor.fetchall()

        conn.close()  # Verbindung schließen

        formatted_data = [{"photo_url": item[0], "shop_url": item[1], "name": item[2]} for item in data]  # Neu strukturierte Daten erstellen
        return Response(content=json.dumps(formatted_data), media_type="application/json")  # Daten als JSON zurückgeben
    except Exception as e:
        return Response(content=json.dumps({"error": str(e)}), media_type="application/json")  # Fehlermeldung zurückgeben
