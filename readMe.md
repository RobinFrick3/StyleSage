
Vorbereitung
Schritt 1: Absolute Dateipfade aktuallisieren -->
    - main.py : 123 & 140
    - classifier_coach : 103
    - classifier : 38
    - sondi_classifier_coach : 103
    - sondi_classifier: 38 - 44
    - database : 6
    - fill_interieur_train_database : 6
    - fill_room_train_database : 6
    - fill_show_database : 6
Schritt 2: Libarys innerhalb der virtuellen Umgebung installieren -->
    - pip install fastapi
    - pip install jinja2
    - pip install Pillow
    - pip install pandas
    - pip install uvicorn
    - install python-multipart
    - pip install torch torchvision


Start
Schritt 1: in Ordner Backend navigieren --> cd .\backend\    
Schritt 3: virtuelle Umgebung starten --> venv\Scripts\activate
Schritt 2: Code ausführen --> uvicorn main:app --reload

Einstellungen
    - main: dotraining & doSondiTraining --> auf true setzen und speichern der server startet wegen der Änderung automatisch neu und bevor die App startet trainiert das Model (Achtung! dauert ein halben Tag und überschreibt die bereits trainierten modelle)
    - main: fill_interieur_train_database, fill_room_train_database, fill_show_database --> Ist für das befüllen der Datenbank (Achtung! sollte die datenbank bereits befüllt sein und es wird trotzdem entkommentiert befinden sich alle Elemente doppelt in der Datenbank)