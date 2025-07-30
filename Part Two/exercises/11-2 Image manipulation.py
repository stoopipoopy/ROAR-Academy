import numpy as np
from matplotlib.image import imread
from PIL import Image

import matplotlib.pyplot as plt

image1 = imread('/Users/petropapahadjopoulos/ROAR-Academy/Part Two/notebooks/lenna.bmp')  
image2 = imread('/Users/petropapahadjopoulos/ROAR-Academy/Part Two/notebooks/airplane.bmp') 
if image1.shape[-1] != image2.shape[-1]:
    raise ValueError("Both images must have the same number of channels.")

rows, cols, _ = image2.shape
image1_height, image1_width, _ = image1.shape

if rows > image1_height or cols > image1_width:
    raise ValueError("The second image is too large to fit in the top-right corner of the first image.")
scale_factor = 0.5  
image2_resized = np.array(Image.fromarray(image2).resize(
    (int(cols * scale_factor), int(rows * scale_factor))
))

rows_resized, cols_resized, _ = image2_resized.shape

if rows_resized > image1_height or cols_resized > image1_width:
    raise ValueError("The resized second image is too large to fit in the top-right corner of the first image.")
image1_copy = image1.copy()  
image1_copy[:rows_resized, -cols_resized:] = image2_resized

plt.imshow(image1_copy)
plt.axis('off')
plt.show()