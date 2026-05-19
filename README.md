# eigenopdracht_LTN
# Cell Cancer Detector

Automatische detectie van borstkanker in microscopische celafbeeldingen met behulp van beeldanalyse en machine learning.

---

##  Projectomschrijving

Dit project analyseert microscopische celafbeeldingen om automatisch onderscheid te maken tussen **normale cellen** en **kankercellen**. Het systeem verwerkt de afbeeldingen stap voor stap: van ruisverwijdering tot segmentatie, feature extractie en uiteindelijk een classificatie via een Random Forest-model.

Het project is ontwikkeld als onderdeel van de skill *Data Science for Biology*, gericht op toepassingen in bio-informatica en medische data-analyse.

---

##  Installatie

### Vereisten
- Python 3.8 of hoger
- pip

### Installeer de benodigde bibliotheken

```bash
pip install opencv-python scikit-image numpy scikit-learn joblib
```

---

##  Gebruik

### 1. Zet je afbeeldingen klaar
Plaats de `.tif`-afbeeldingen die je wilt analyseren in de map `data/test/`.

### 2. Train het model
Zorg dat de trainingsdata beschikbaar is in `data/train/` met bijbehorende labels, en voer dan uit:

```bash
python train.py
```

### 3. Analyseer nieuwe afbeeldingen
```bash
python predict.py
```

De uitvoer ziet er zo uit:

```
OVERZICHT:
afbeelding1.tif : normal
afbeelding2.tif : cancer
```

---

##  Hoe werkt het?

```
Afbeelding (.tif)
      │
      ▼
Preprocessing        →  Ruis verwijderen (Gaussian Blur) + drempelwaarde (Otsu)
      │
      ▼
Segmentatie          →  Elke cel krijgt een uniek label
      │
      ▼
Feature Extractie    →  Oppervlakte, omtrek, intensiteit, vorm, textuur (GLCM)
      │
      ▼
Classificatie        →  Random Forest voorspelt: normaal of kanker
      │
      ▼
Resultaat            →  "normal" / "cancer" per afbeelding
```

### Gebruikte kenmerken per cel

| Kenmerk | Beschrijving |
|---|---|
| `area` | Oppervlakte van de cel in pixels |
| `perimeter` | Omtrek van de cel |
| `intensity_mean` | Gemiddelde pixelintensiteit |
| `eccentricity` | Mate van ellipsvorm (0 = cirkel, 1 = lijn) |
| `solidity` | Compactheid van de cel |
| `shape_irregularity` | Hoe onregelmatig de celrand is |
| `contrast` | Textuurcontrast (GLCM) |
| `homogeneity` | Textuurhomogeniteit (GLCM) |
| `size_variance` | Variantie in celgrootte binnen de afbeelding |

---

##  Projectstructuur

```
project/
│
├── data/
│   ├── train/          # Trainingsafbeeldingen met labels
│   └── test/           # Testafbeeldingen voor voorspelling
│
├── train.py            # Model trainen op de dataset
├── predict.py          # Nieuwe afbeeldingen analyseren
├── features.py         # Feature extractie logica
├── model.pkl           # Opgeslagen getraind model
└── README.md
```

---

##  Gebruikte bibliotheken

| Bibliotheek | Gebruik |
|---|---|
| `OpenCV` | Afbeelding inladen en preprocessing |
| `scikit-image` | Segmentatie, feature extractie en GLCM |
| `NumPy` | Numerieke berekeningen |
| `scikit-learn` | Random Forest classificatie en evaluatie |
| `joblib` | Model opslaan en inladen |

---

##  Evaluatie

Het model wordt geëvalueerd met een classificatierapport dat precisie, recall en F1-score weergeeft per klasse:

```
              precision    recall  f1-score

      normal       0.XX      0.XX      0.XX
      cancer       0.XX      0.XX      0.XX
```

---

##  Auteur

**Luuk**
Opleiding: Biologie / Bio-informatica
Project: Data Science for Biology – Celidentificatie met Beeldanalyse

---

## 📄 Licentie

Dit project is gemaakt voor educatieve doeleinden.
