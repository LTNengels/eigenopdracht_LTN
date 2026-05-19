from skimage import data, io
import os
os.makedirs("../raw/raw", exist_ok=True)
image = data.cells3d()[30, 1]
io.imsave("../raw/cell_slice.tif", image)