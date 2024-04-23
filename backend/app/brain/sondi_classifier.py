import torch  # Importieren der torch-Bibliothek für PyTorch
import torch.nn as nn  # Importieren der nn-Module von torch für neuronale Netzwerke
from torch.utils.data import Dataset  # Importieren des Dataset-Moduls von torch für benutzerdefinierte Datensätze
import torchvision.transforms as transforms  # Importieren von transforms aus torchvision für Bildtransformationen
from PIL import Image  # Importieren der Image-Klasse aus dem PIL-Modul für die Bildverarbeitung
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
        image = Image.open(image_path).convert("RGB")  # Öffnen des Bildes und Konvertieren in das RGB-Format
        if self.transform:
            image = self.transform(image)  # Anwenden der Transformationen auf das Bild
        return image, label  # Rückgabe des transformierten Bildes und des Labels

# Modell definieren
class StyleClassifier(nn.Module):
    def __init__(self, num_classes):
        super(StyleClassifier, self).__init__()  # Aufruf der __init__-Methode der Elternklasse
        self.resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)  # Laden des ResNet-Modells
        num_features = self.resnet.fc.in_features  # Anzahl der Eingangsmerkmale des letzten vollständig verbundenen Layers
        self.resnet.fc = nn.Linear(num_features, num_classes)  # Ersetzen des letzten Layers durch einen neuen vollständig verbundenen Layer

    def forward(self, x):
        x = self.resnet(x)  # Durchführen der Vorwärtsdurchlaufoperation
        return x  # Rückgabe der Ausgabe

# Pfade zu den Modellen definieren
model_paths = {
    'sofa_model': 'C:\\Users\\I551667\\Documents\\GitHub\\StyleSage-Interieur\\backend\\sofa_collection_model.pth',
    'tisch_model': 'C:\\Users\\I551667\\Documents\\GitHub\\StyleSage-Interieur\\backend\\tisch_collection_model.pth',
    'schrank_model': 'C:\\Users\\I551667\\Documents\\GitHub\\StyleSage-Interieur\\backend\\schrank_collection_model.pth',
    'bett_model': 'C:\\Users\\I551667\\Documents\\GitHub\\StyleSage-Interieur\\backend\\bett_collection_model.pth',
    'stuhl_model': 'C:\\Users\\I551667\\Documents\\GitHub\\StyleSage-Interieur\\backend\\stuhl_collection_model.pth',
    'beleuchtung_model': 'C:\\Users\\I551667\\Documents\\GitHub\\StyleSage-Interieur\\backend\\beleuchtung_collection_model.pth',
    'deko_model': 'C:\\Users\\I551667\\Documents\\GitHub\\StyleSage-Interieur\\backend\\deko_collection_model.pth',
}

# Funktion zum Laden des trainierten Modells und Klassifizieren eines Bildes
def classify_image_from_file(image_file, model_label):
    model_path = model_paths.get(model_label)  # Abrufen des Modellpfads anhand des Labels
    if model_path is None:
        print(f"Das Modell '{model_label}' wurde nicht gefunden.")
        return None

    num_classes = 6  # Anzahl der Klassen (änderbar)

    # Modell laden
    model = StyleClassifier(num_classes)  # Instanziierung des Modells
    model.load_state_dict(torch.load(model_path))  # Laden der Modellparameter
    model.eval()  # Wechseln in den Evaluierungsmodus

    # Bild klassifizieren
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Ändern der Bildgröße auf 224x224
        transforms.ToTensor(),  # Konvertieren des Bildes in einen Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalisieren der Bildpixel
    ])
    image = Image.open(image_file).convert("RGB")  # Öffnen des Bildes und Konvertieren in das RGB-Format
    image = transform(image).unsqueeze(0)  # Anwenden der Transformationen und Hinzufügen einer zusätzlichen Dimension für den Batch
    with torch.no_grad():
        output = model(image)  # Durchführen der Vorwärtsdurchlaufoperation
    _, predicted = torch.max(output, 1)  # Ermitteln der vorhergesagten Klasse
    return predicted.item() + 1  # Rückgabe der vorhergesagten Klasse (mit Anpassung um 1 für die Originalklassen)
