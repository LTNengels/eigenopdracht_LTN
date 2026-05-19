import os
import cv2
from skimage import filters
from skimage import io

# dir

input_path = r"C:\Users\luuke\PycharmProjects\PythonProject\raw\cell_slice.tif"
output_dir = r"C:\Users\luuke\PycharmProjects\PythonProject\raw"

os.makedirs(output_dir, exist_ok=True)

# inlezen
image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError(f"Image niet gevonden: {input_path}")

#blur
blurred = cv2.GaussianBlur(image, (5, 5), 0)

#treshold
thresh = filters.threshold_otsu(blurred)
binary = blurred > thresh

# -------------------------
# opslaan
# -------------------------
cv2.imwrite(os.path.join(output_dir, "blurred.tif"), blurred)
io.imsave(os.path.join(output_dir, "binary.tif"), (binary.astype("uint8") * 255))

print("Processing klaar!")
print("threshold:", thresh)