from ..brain.sondi_classifier import classify_image_from_file  # Importieren der Funktion zum Klassifizieren von Bildern aus dem Sondi-Modul
from collections import Counter  # Importieren der Counter-Klasse aus dem collections-Modul

# Definieren von Schlüsselwörtern für verschiedene Möbelkategorien
sofa_keywords = ['couch']
schrank_keywords = ['wardrobe', 'sideboard', 'cupboard', 'closet']
deko_keywords = ['plant', 'clock', 'pottedplant', 'vase', 'bowl', 'book', 'bottle']
beleuchtung_keywords = ['lamp', 'sofa']
bett_keywords = ['bed']
tisch_keywords = ['table', 'diningtable']
stuhl_keywords = ['chair']

# Funktion zur Berechnung des Stils anhand von Bildern
def calculate_Style(image_data):
    style_list = []
    # Durchlaufen der Bilddaten
    for image_dict in image_data:
        if 'image_link' in image_dict:
            image_path = image_dict['image_link']
            if 'label' in image_dict:
                image_label = image_dict['label']
                interieur_model = get_interieur_model(image_label)
                if interieur_model is not None:
                    style_list.append((interieur_model, classify_image_from_file(image_path, interieur_model)))
    return get_final_style(style_list)  # Rückgabe des finalen Stils

# Funktion zum Zuordnen von Bildetiketten zu Möbelkategorien
def get_interieur_model(image_label):
    if any(keyword in image_label.lower() for keyword in sofa_keywords):
        return 'sofa_model'
    elif any(keyword in image_label.lower() for keyword in schrank_keywords):
        return 'schrank_model'
    elif any(keyword in image_label.lower() for keyword in bett_keywords):
        return 'bett_model'
    elif any(keyword in image_label.lower() for keyword in tisch_keywords):
        return 'tisch_model'
    elif any(keyword in image_label.lower() for keyword in stuhl_keywords):
        return 'stuhl_model'
    elif any(keyword in image_label.lower() for keyword in beleuchtung_keywords):
        return 'beleuchtung_model'
    elif any(keyword in image_label.lower() for keyword in deko_keywords):
        return 'deko_model'
    return None  # Rückgabe von 'None', falls keine passende Möbelkategorie gefunden wurde

# Funktion zur Ermittlung des endgültigen Stils basierend auf den ermittelten Stilen der Bildkategorien
def get_final_style(style_list):
    if not style_list:
        return None
    
    # Separate Zähler für jede Möbelkategorie
    counter_dict = {
        'sofa_model': Counter(),
        'schrank_model': Counter(),
        'bett_model': Counter(),
        'tisch_model': Counter(),
        'stuhl_model': Counter(),
        'beleuchtung_model': Counter(),
        'deko_model': Counter()
    }

    # Fülle die Zähler mit den entsprechenden Stilen
    for model, style in style_list:
        if model in counter_dict:
            counter_dict[model][style] += 1
    
    # Finde die am häufigsten vorkommenden Stile für jede Möbelkategorie
    most_common_styles = {}
    for model, count in counter_dict.items():
        if count:
            most_common_styles[model] = count.most_common(1)[0][0]

    
    # Finde den häufigsten Stil unter den am häufigsten vorkommenden Stilen
    final_style_counter = Counter(most_common_styles.values())
    most_common_final_style = final_style_counter.most_common(1)[0][0]
    
    return most_common_final_style  # Rückgabe des endgültigen Stils
