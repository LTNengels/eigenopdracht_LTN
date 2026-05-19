import os
import cv2
import numpy as np

from skimage import filters
from skimage.measure import regionprops, label
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# -------------------------
# 1. DATA PAD
# -------------------------
data_dir = "raw2"

features = []
labels = []

# -------------------------
# 2. LOOP DOOR FOLDERS
# -------------------------
for class_name in ["normal", "cancer"]:

    folder = os.path.join(data_dir, class_name)

    if class_name == "cancer":
        image_label = 1
    else:
        image_label = 0

    for file in os.listdir(folder):

        path = os.path.join(folder, file)

        # -------------------------
        # 3. IMAGE LOAD
        # -------------------------
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            continue

        # -------------------------
        # 4. PREPROCESSING
        # -------------------------
        blur = cv2.GaussianBlur(image, (5, 5), 0)

        thresh = filters.threshold_otsu(blur)
        binary = blur > thresh

        # -------------------------
        # 5. SEGMENTATIE
        # -------------------------
        label_img = label(binary)

        # -------------------------
        # 6. FEATURE EXTRACTION
        # -------------------------
        from skimage.feature import graycomatrix, graycoprops

        # extra: nucleus size variance per image
        regions_all = list(regionprops(label_img, intensity_image=blur))
        areas = [r.area for r in regions_all if r.area >= 50]

        size_variance = np.var(areas) if len(areas) > 0 else 0

        for region in regions_all:

            if region.area < 50:
                continue

            # -------------------------
            # SHAPE FEATURES
            # -------------------------
            area = region.area
            perimeter = region.perimeter

            if perimeter > 0:
                shape_irregularity = (perimeter ** 2) / (4 * np.pi * area)
            else:
                shape_irregularity = 0

            eccentricity = region.eccentricity
            solidity = region.solidity

            # -------------------------
            # INTENSITY
            # -------------------------
            intensity = region.intensity_mean

            # -------------------------
            # TEXTURE (GLCM)
            # -------------------------
            minr, minc, maxr, maxc = region.bbox
            patch = blur[minr:maxr, minc:maxc]

            if patch.size == 0:
                contrast = 0
                homogeneity = 0
            else:
                patch = patch.astype(np.uint8)

                glcm = graycomatrix(
                    patch,
                    distances=[1],
                    angles=[0],
                    levels=256,
                    symmetric=True,
                    normed=True
                )

                contrast = graycoprops(glcm, 'contrast')[0, 0]
                homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]

            # -------------------------
            # FINAL FEATURE VECTOR
            # -------------------------
            f = [
                area,
                perimeter,
                intensity,
                eccentricity,
                solidity,
                shape_irregularity,
                contrast,
                homogeneity,
                size_variance
            ]

            features.append(f)
            labels.append(image_label)
# -------------------------
# 7. DATASET
# -------------------------
X = np.array(features)
y = np.array(labels)

print("Totaal samples:", len(X))

# -------------------------
# 8. TRAIN / TEST SPLIT
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# 9. MACHINE LEARNING
# -------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -------------------------
# 10. EVALUATIE
# -------------------------
y_pred = model.predict(X_test)

print("\nCLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred))

# -------------------------
# 11. VOORBEELD OUTPUT
# -------------------------
print("\nVoorbeeld voorspellingen:", y_pred[:10])

import joblib
joblib.dump(model, "model2.pkl")