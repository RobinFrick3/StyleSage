let CalculatedStyle;

async function calculateStyleCategory() {
    try {
        
        // Daten basierend auf Style ID abrufen
        const styleResponse = await fetch('/style_category');  // Abrufen der Style-Kategorie-Daten vom Server
        const styleData = await styleResponse.json();  // Konvertieren der Serverantwort in JSON-Format

        // Stylekategorie und ID abrufen
        CalculatedStyle = styleData.style_id;  // Extrahieren der Style-ID aus den Daten

        if (CalculatedStyle === null) {  // Wenn keine Style-Kategorie gefunden wurde
            // Überschrift anpassen
            document.querySelector('.galleryHead h2').textContent = "Tut mir leid, auf deinem Foto konnte keine Inneneinrichtung erkannt werden";  // Überschrift anpassen

            // Beende die Funktion, keine weiteren Kacheln anzeigen
            return;
        }

        // Überschrift anpassen
        document.querySelector('.galleryHead h2').textContent = styleData.style_name;

        await filterFurniture();  // Filtern der Möbel beim Laden der Seite

    } catch (error) {
        console.error('Fehler beim Laden der Galerie:', error);  // Fehlerbehandlung für Fehler beim Laden der Galerie
    }
}

// Funktion, um die Möbel nach der ausgewählten Möbelart zu filtern
async function filterFurniture() {
    const selectedType = document.getElementById('furnitureType').value;  // Abrufen des ausgewählten Möbeltyps aus dem Dropdown-Menü
    const gallery = document.getElementById('gallery');
    gallery.innerHTML = ''; // Leert die Galerie, um sie neu zu erstellen

    try {

        // Daten basierend auf Möbelart und Style ID abrufen
        const url = selectedType !== "0" ? `/get_data_from_database?furniture_type=${selectedType}&style_type=${CalculatedStyle}` : `/get_data_from_database?style_type=${CalculatedStyle}`;  // Aufbau der URL für den Datenabruf basierend auf dem ausgewählten Möbeltyp und der Style-ID
        
        // Daten abrufen
        const furnitureResponse = await fetch(url);  // Abrufen der Möbel-Daten vom Server
        const furnitureData = await furnitureResponse.json();  // Konvertieren der Serverantwort in JSON-Format

        // Galerie aktualisieren
        furnitureData.forEach(item => {
            const tile = document.createElement('div');  // Erstellen eines Kachel-Elements für jedes Möbelstück
            tile.classList.add('tile');  // Hinzufügen der CSS-Klasse 'tile' zum Kachel-Element

            // Bild hinzufügen
            const image = document.createElement('img');  // Erstellen eines Bild-Elements für das Möbelstück
            image.src = item.photo_url;  // Festlegen der Bildquelle basierend auf der URL des Möbelfotos
            tile.appendChild(image);  // Hinzufügen des Bildes zum Kachel-Element

            // Name hinzufügen
            const name = document.createElement('p');  // Erstellen eines Paragraphen-Elements für den Möbelnamen
            name.textContent = item.name;  // Festlegen des Textinhalts des Paragraphen basierend auf dem Möbelnamen
            tile.appendChild(name);  // Hinzufügen des Paragraphen zum Kachel-Element

            // Klickereignis hinzufügen
            tile.addEventListener('click', () => {
                window.location.href = item.shop_url; // Auf Shopseite navigieren, wenn auf das Möbelstück geklickt wird
            });

            gallery.appendChild(tile);  // Hinzufügen des Kachel-Elements zur Galerie
        });
    } catch (error) {
        console.error('Fehler beim Laden der Galerie:', error);  // Fehlerbehandlung für Fehler beim Laden der Galerie
    }
}

// Funktion zum Laden des hochgeladenen Bildes
function loadUploadedImage() {
    fetch('/temp_image')  // Abrufen des temporären Bildes vom Server
    .then(response => response.blob())  // Konvertieren der Serverantwort in einen Blob
    .then(blob => {
        var objectURL = URL.createObjectURL(blob);  // Erzeugen einer URL für das Blob-Objekt
        document.getElementById('uploadedImage').src = objectURL;  // Festlegen der Bildquelle des Bild-Elements auf die erzeugte URL
    })
    .catch(error => console.error('Fehler:', error));  // Fehlerbehandlung für Fehler beim Laden des Bildes
}

// Wird aufgerufen, sobald die Seite geladen ist
window.onload = () => {
    loadUploadedImage();  // Laden des hochgeladenen Bildes
    calculateStyleCategory();
}