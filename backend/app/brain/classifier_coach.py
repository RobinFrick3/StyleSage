import torch  # Importieren der torch-Bibliothek für PyTorch
import torch.nn as nn  # Importieren der nn-Module von torch für neuronale Netzwerke
import torchvision.models as models  # Importieren von Modellen aus torchvision für Computer Vision
import torchvision.transforms as transforms  # Importieren von transforms aus torchvision für Bildtransformationen
from PIL import Image  # Importieren der Image-Klasse aus dem PIL-Modul für die Bildverarbeitung
from torch.utils.data import DataLoader, Dataset  # Importieren von DataLoader und Dataset aus torch.utils.data für das Laden von Daten
import sqlite3  # Importieren des sqlite3-Moduls für die Interaktion mit SQLite-Datenbanken
import os  # Importieren des os-Moduls zur Interaktion mit dem Betriebssystem

# Definition der benutzerdefinierten Dataset-Klasse
class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data  # Zuweisung der Daten an das Attribut self.data
        self.transform = transform  # Zuweisung der Transformationen an das Attribut self.transform

    def __len__(self):
        return len(self.data)  # Rückgabe der Länge des Datensatzes

    def __getitem__(self, idx):
        image_path, label = self.data[idx]  # Abrufen des Bildpfads und des Labels
        try:
            image = Image.open(image_path).convert("RGB")  # Öffnen des Bildes und Konvertieren in das RGB-Format
            if self.transform:
                image = self.transform(image)  # Anwenden der Transformationen auf das Bild
            return image, label  # Rückgabe des transformierten Bildes und des Labels
        except Exception as e:
            print(f"Error opening image {os.path.basename(image_path)}")
            return self.__getitem__((idx + 1) % len(self.data))  # Erneuter Versuch mit dem nächsten Bild

# Modell definieren
class StyleClassifier(nn.Module):
    def __init__(self, num_classes):
        super(StyleClassifier, self).__init__()  # Aufruf der __init__-Methode der Elternklasse
        self.resnet = models.resnet18(pretrained=False)  # Laden des ResNet-Modells mit pretrained=False
        num_features = self.resnet.fc.in_features  # Anzahl der Eingangsmerkmale des letzten vollständig verbundenen Layers
        self.resnet.fc = nn.Linear(num_features, num_classes)  # Ersetzen des letzten Layers durch einen neuen vollständig verbundenen Layer

    def forward(self, x):
        x = self.resnet(x)  # Durchführen der Vorwärtsdurchlaufoperation
        return x  # Rückgabe der Ausgabe

def train_model(data, model_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Ändern der Bildgröße auf 224x224
        transforms.ToTensor(),  # Konvertieren des Bildes in einen Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalisieren der Bildpixel
    ])

    # DataLoader zum Laden der Daten in Batches erstellen
    batch_size = 32
    dataset = CustomDataset(data=data, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Anzahl der Style-Kategorien (1 bis 6)
    num_classes = 6

    # Modell initialisieren
    model = StyleClassifier(num_classes)

    # Verlustfunktion, Optimierer usw. definieren
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Modell trainieren
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    num_epochs = 10

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for i, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)

            # Berechnung des Fortschritts in Prozent
            progress = (i + 1) / len(dataloader) * 100
            print(f'Epoch [{epoch+1}/{num_epochs}], Progress: {progress:.2f}%', end='\r')

        epoch_loss = running_loss / len(dataset)
        epoch_accuracy = correct_predictions / total_predictions
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')

    # Speichern der trainierten Gewichte
    torch.save(model.state_dict(), model_path)

# Datenbankabfrage
def fetch_data_from_database():
    conn = sqlite3.connect('C:\\Users\\I551667\\Documents\\GitHub\\StyleSage-Interieur\\backend\\app\\Datenbank\\StyleSage.db')
    cursor = conn.cursor()
    cursor.execute("SELECT photo_name, style_id FROM room_photos")
    data = cursor.fetchall()
    conn.close()

    # Bildpfade erstellen und der Datenliste hinzufügen
    data_with_image_path = [(create_image_path(photo_name), style_id - 1) for photo_name, style_id in data]  # Reduzieren Sie das Label um 1
    return data_with_image_path

# Funktion zum Erstellen des Dateipfads zu den Bildern
def create_image_path(photo_name):
    database_dir = os.path.dirname(os.path.abspath(__file__))  # Verzeichnis dieser Datei (main.py)
    pics_dir = os.path.join(database_dir, '..', 'Datenbank', 'pics', 'room_pics')  # Verzeichnis für die Bilder
    image_path = os.path.join(pics_dir, photo_name)  # Dateipfad zum Bild
    return image_path

# Hauptfunktion
if __name__ == "__main__":
    data = fetch_data_from_database()  # Daten aus der Datenbank abrufen
    model_path = 'model.pth'  # Pfad zum speichern des trainierten Modells
    train_model(data, model_path)  # Modell trainieren
