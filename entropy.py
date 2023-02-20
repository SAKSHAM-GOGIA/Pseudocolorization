import matplotlib.pyplot as plt
from skimage import io
img = io.imread("./images/landscape_original.jpg") #path to the image
plt.imshow(img)
plt.show()
print(img.shape) # show dimension of image
dim1, dim2 = img.shape[0], img.shape[1]
num_channels = img.shape[2]

plt.imshow(img[:,:,0])

from skimage.transform import rescale, resize
def resized_img(img, scale=2):
    image_resized = resize(img, (img.shape[0] / scale, img.shape[1] / scale),anti_aliasing=True)
    return image_resized

img = io.imread("./images/landscape_grey.jpg")
image_resized = resized_img(img, 2) # I choosed 2, but you could all integers
def plot_compared_img(img1, img2, title1="Original image", title2="New one image"):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
    ax = axes.ravel()
    ax[0].imshow(img1)
    ax[0].set_title(title1)
    ax[1].imshow(img2, cmap='gray')
    ax[1].set_title(title2)
    plt.show()
    
plot_compared_img(img, image_resized,"Original image", "Resized image")

import numpy as np
k = 0.2 # you could set any any real number
noise = np.ones_like(img) * k * (img.max() - img.min())
noise[np.random.random(size=noise.shape) > 0.5] *= -1
img_noise = img + noise # new image with noise
plot_compared_img(img, img_noise, "Original image", "The image with noise")


from skimage.metrics import structural_similarity as ssim
ssim_noise = ssim(img, img_noise,
                  data_range=img_noise.max() - img_noise.min(), multichannel=True)
print(ssim_noise) #0.10910931410753069

from skimage.color import rgb2gray
grayscale = rgb2gray(img)
plot_compared_img(img, grayscale,"Original image", "The image with greyscale")

from skimage.measure.entropy import shannon_entropy
print(shannon_entropy(img[:,:,0])) #7.5777861360050265
#plot entropy
from skimage.filters.rank import entropy
from skimage.morphology import disk
entr_img = entropy(img[:,:,0], disk(10))
plt.imshow(entr_img, cmap='viridis')
plt.show()