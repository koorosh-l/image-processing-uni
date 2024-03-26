import matplotlib.pyplot as plt

from skimage import data
from skimage.color import rgb2gray

original_pic = data.cat()
grayscaled_pic = rgb2gray(original_pic)

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
ax = axes.ravel()

ax[0].imshow(original_pic)
ax[0].set_title("Original Picture")
ax[1].imshow(grayscaled_pic, cmap=plt.cm.gray)
ax[1].set_title("Grayscaled Picture")

fig.tight_layout()
plt.show()