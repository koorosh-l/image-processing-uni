import skimage
from skimage import io, color

image = io.imread('fox.jpeg')
gray_image = color.rgb2gray(image)

io.imshow(image)
io.show()