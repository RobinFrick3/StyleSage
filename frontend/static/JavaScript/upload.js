// Event Listener für den Bereich für das Drag & Drop-Hochladen
var dropZone = document.getElementById('drop_zone');
dropZone.addEventListener('dragover', handleDragOver, false); // Event Listener für das Dragover-Ereignis hinzufügen
dropZone.addEventListener('drop', handleFileSelect, false); // Event Listener für das Drop-Ereignis hinzufügen

function handleDragOver(event) {
    event.stopPropagation(); // Stoppt die weitere Ausbreitung des Ereignisses
    event.preventDefault(); // Verhindert das Standardverhalten des Browsers (z.B. das Öffnen der Datei im Browserfenster)
    event.dataTransfer.dropEffect = 'copy'; // Zeigt den Cursor als "Kopieren" beim Drag & Drop
}

function handleFileSelect(event) {
    event.stopPropagation(); // Stoppt die weitere Ausbreitung des Ereignisses
    event.preventDefault(); // Verhindert das Standardverhalten des Browsers (z.B. das Öffnen der Datei im Browserfenster)
    var files = event.dataTransfer.files; // Dateien aus dem Drop-Ereignis extrahieren
    var fileInput = document.getElementById('fileinput');
    fileInput.files = files; // Die ausgewählten Dateien dem unsichtbaren Datei-Eingabefeld zuweisen
}

// Event Listener für den Upload-Button
document.getElementById('uploadButton').addEventListener('click', function() {
    var fileInput = document.getElementById('fileinput'); // Das Datei-Eingabefeld abrufen
    var formData = new FormData(); // Neue FormData-Instanz erstellen
    formData.append('imageFile', fileInput.files[0]); // Die ausgewählte Datei der FormData hinzufügen

    fetch('/Upload', { // Datei an den Server hochladen
        method: 'POST', // HTTP-Methode festlegen
        body: formData // Formulardaten als Körper der Anfrage senden
    })
    .then(response => {
        if (response.redirected) { // Überprüfen, ob eine Weiterleitung vom Server empfangen wurde
            window.location.href = response.url; // Weiterleitung zur empfangenen URL
        }
    })
    .catch(error => console.error('Error:', error)); // Fehlerbehandlung für Fehler beim Hochladen der Datei
});
