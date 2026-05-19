import os
import cv2
import numpy as np
import joblib

from skimage import filters
from skimage.measure import regionprops, label
from skimage.feature import graycomatrix, graycoprops

# -------------------------
# MODEL LADEN
# -------------------------
model = joblib.load("model2.pkl")

# -------------------------
# FOLDER MET IMAGES
# -------------------------
folder_path = r"C:\Users\luuke\PycharmProjects\PythonProject\scrips\raw2/cancer"

results = []

# -------------------------
# LOOP DOOR ALLE IMAGES
# -------------------------
for file in os.listdir(folder_path):

    image_path = os.path.join(folder_path, file)

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        continue

    # -------------------------
    # PREPROCESSING
    # -------------------------
    blur = cv2.GaussianBlur(image, (5, 5), 0)

    thresh = filters.threshold_otsu(blur)
    binary = blur > thresh

    label_img = label(binary)

    # -------------------------
    # FEATURE EXTRACTION
    # -------------------------
    features = []

    regions = list(regionprops(label_img, intensity_image=blur))

    areas = [r.area for r in regions if r.area > 50]
    size_variance = np.var(areas) if len(areas) > 0 else 0

    for region in regions:

        if region.area < 50:
            continue

        area = region.area
        perimeter = region.perimeter

        shape_irregularity = (perimeter**2) / (4*np.pi*area) if area > 0 else 0

        intensity = region.intensity_mean
        eccentricity = region.eccentricity
        solidity = region.solidity

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

    # -------------------------
    # PREDICTIE
    # -------------------------
    if len(features) == 0:
        continue

    X = np.array(features)

    preds = model.predict(X)

    final = int(np.round(np.mean(preds)))

    # -------------------------
    # RESULTAAT OPSLAAN
    # -------------------------
    if final == 1:
        result = "cancer"
    else:
        result = "normal"

    results.append((file, result))

    print(file, "→", result)

# -------------------------
# SAMENVATTING
# -------------------------
print("\nOVERZICHT:")
for r in results:
    print(r[0], ":", r[1])