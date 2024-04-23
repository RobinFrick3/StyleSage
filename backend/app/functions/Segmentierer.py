import pandas as pd  # Importieren der pandas-Bibliothek zur Datenmanipulation
import urllib.request  # Importieren des Moduls urllib.request zur Arbeit mit URLs
import json  # Importieren der json-Bibliothek zur Verarbeitung von JSON-Daten
import os  # Importieren des os-Moduls zur Interaktion mit dem Betriebssystem
import ssl  # Importieren des ssl-Moduls zur Verwendung von HTTPS-Verbindungen
import base64  # Importieren der base64-Bibliothek zur Kodierung von Daten
from PIL import Image  # Importieren der Image-Klasse aus dem PIL-Modul für die Bildverarbeitung
import tempfile  # Importieren des tempfile-Moduls zum Erstellen temporärer Dateien

# Funktion zur Erlaubnis von selbstsignierten HTTPS-Verbindungen
def allowSelfSignedHttps(allowed):
    # Bypass der Serverzertifikatsprüfung auf der Clientseite
    if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context

# Funktion zum Segmentieren eines Bildes
def segment_image(image_path):
    allowSelfSignedHttps(True)  # Diese Zeile ist erforderlich, wenn ein selbstsigniertes Zertifikat im Bereitstellungsdienst verwendet wird.

    # Lesen der Bilddatei
    with open(image_path, 'rb') as image_file:
        image_bytes = image_file.read()
        # Konvertieren der Bilddaten in eine base64-kodierte Zeichenfolge
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')

    # Erstellen eines DataFrame direkt aus der base64-kodierten Bilddatei
    data = {
        "input_data": {
            "columns": ["image"],
            "index": [0],
            "data": [image_base64]
        },
        "params": {}
    }

    body = str.encode(json.dumps(data))

    url = 'https://robin-frick-sondi-projekt-fqmir.westeurope.inference.ml.azure.com/score'  # URL des Bereitstellungsdienstes
    api_key = 'c02Y3WFbRevTgkpTAX7mxJz4nWPAIqci'  # API-Schlüssel für den Zugriff auf den Dienst
    if not api_key:
        raise Exception("A key should be provided to invoke the endpoint")

    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + api_key,
        'azureml-model-deployment': 'automl-image-instance-segment-3'
    }

    req = urllib.request.Request(url, body, headers)

    cropped_images_info = []  # Liste zur Speicherung von Informationen über zugeschnittene Bilder (Link und Beschriftung)

    try:
        response = urllib.request.urlopen(req)
        result = response.read()
        # Dekodieren des JSON-Ergebnisses
        result_json = json.loads(result)
        
        # Abrufen der Bildabmessungen
        image = Image.open(image_path)
        image_width, image_height = image.size
        
        # Durchlaufen jedes Wörterbuchs in der result_json-Liste
        for result_dict in result_json:
            if 'boxes' in result_dict:
                boxes_list = result_dict['boxes']
                # Durchlaufen jedes Felds in der boxes_liste
                for box_dict in boxes_list:
                    if 'box' in box_dict:
                        box = box_dict['box']
                        # Umrechnen der relativen Koordinaten in Pixelkoordinaten
                        topX, topY = int(box['topX'] * image_width), int(box['topY'] * image_height)
                        bottomX, bottomY = int(box['bottomX'] * image_width), int(box['bottomY'] * image_height)
                        
                        # Zuschneiden des Bildes
                        cropped_image = image.crop((topX, topY, bottomX, bottomY))
                        
                        # Speichern des zugeschnittenen Bildes vorübergehend
                        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                        cropped_image.save(temp_file.name)
                        
                        # Abrufen der Beschriftung
                        label = box_dict['label']
                        # Anfügen des Bildlinks und der Beschriftung an die Liste
                        cropped_images_info.append({"image_link": temp_file.name, "label": label})

    except urllib.error.HTTPError as error:
        print("The request failed with status code: " + str(error.code))
        print(error.info())
        print(error.read().decode("utf8", 'ignore'))

    return cropped_images_info  # Rückgabe der Informationen über die zugeschnittenen Bilder
