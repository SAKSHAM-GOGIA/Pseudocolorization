import numpy as np 
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from skimage import io

print("loading models.....")
net = cv2.dnn.readNetFromCaffe('./model/colorization_deploy_v2.prototxt','./model/colorization_release_v2.caffemodel')
pts = np.load('./model/pts_in_hull.npy')
#  The values in pts_in_hull.npy are the 313 cluster kernels that you can compute out of just stacking a bunch of a and b values in 2-D space.

class8 = net.getLayerId("class8_ab")    # Prototxt file in the end!
conv8 = net.getLayerId("conv8_313_rh")  # activation layer- prediction of values, multinomial probability distribution
pts = pts.transpose().reshape(2,313,1,1)

net.getLayer(class8).blobs = [pts.astype("float32")]  # converting pts in float
net.getLayer(conv8).blobs = [np.full([1,313],2.606,dtype='float32')] # initialising array with fill_value as 2.606 (blobs is a communication method using array)

orig = cv2.imread('./images/imag/1.jpg')
image = cv2.imread('./images/imag/1g.jpg')  


scaled = image.astype("float32")/255.0  # scaled in the range of 0-1
lab = cv2.cvtColor(scaled,cv2.COLOR_BGR2LAB) #


# Conversion of BGR to LAB components using mathematical caluclations 
# L – Lightness ( Intensity ).
# a – color component ranging from Green to Magenta.
# b – color component ranging from Blue to Yellow.

resized = cv2.resize(lab,(224,224)) 
L = cv2.split(resized)[0] 
L -= 50

net.setInput(cv2.dnn.blobFromImage(L))
ab = net.forward()[0, :, :, :].transpose((1,2,0))

ab = cv2.resize(ab, (image.shape[1],image.shape[0]))

L = cv2.split(lab)[0]
colorized = np.concatenate((L[:,:,np.newaxis], ab), axis=2)  #concatinating pixels of colourized image to the array

colorized = cv2.cvtColor(colorized,cv2.COLOR_LAB2BGR)
colorized = np.clip(colorized,0,1) # scaling to values in 0-1 range

colorized = (255 * colorized).astype("uint8")

# Image showed
cv2.imshow("Grey Scale",image)
cv2.imshow("Colorized",colorized)
cv2.imshow("Original", orig)

# load the input images
img1 = orig
img2 = colorized

def mse(img1, img2):
    dim = img1.shape   # tuple -> height, width
    h = dim[0]
    w = dim[1]
    diff = cv2.subtract(img1, img2)
    # print(diff)
    err = np.sum(diff**2)
    mse = err/(float(h*w))
    msre = np.sqrt(mse)
    return mse, diff, msre
    
# define the function to compute MSE between two images
error, diff, msre= mse(img1, img2)
print("Image matching Error between the two images:", round(error, 6))


from math import log10, sqrt

def PSNR(original, colored):
    mse = np.mean((original - colored) ** 2)
    # print(original - colored)
    if(mse == 0):  # MSE is zero means no noise is present in the signal. Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return mse, psnr

mean_sq_er, psnr =  PSNR(orig, colorized)
print("Mean Square Error between the images is:", round(mean_sq_er, 6))
print(f"Peak signal-to-noise ratio between the images is: ", round(psnr, 6))

from skimage.metrics import structural_similarity as ssim
im1 = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
im2 = cv2.cvtColor(colorized, cv2.COLOR_BGR2GRAY)
s = ssim(im1, im2)
print("Structural Similarity Index between the images is:", round(s, 6))

difference = cv2.subtract(img1, img2)

fig = plt.figure(figsize=(20, 14))

fig.add_subplot(2, 3, 1)
plt.imshow(orig)
plt.title('Original Image')
plt.axis('off')

fig.add_subplot(2, 3, 2)
plt.imshow(image)
plt.title('Grey Scale Image')
plt.axis('off')

fig.add_subplot(2, 3, 4)
plt.imshow(colorized)
plt.title('Colorized Image')
plt.axis('off')

fig.add_subplot(2, 3, 5)
plt.imshow(difference)
plt.title('Difference In Image')
plt.axis('off')

from skimage.measure.entropy import shannon_entropy
print("Entropy of original Image is: ", end="")
print(round(shannon_entropy(orig[:,:,0]), 6))
#plot entropy
from skimage.filters.rank import entropy
from skimage.morphology import disk
entr_img1 = entropy(orig[:,:,0], disk(10))
fig.add_subplot(2, 3, 3)
plt.imshow(entr_img1, cmap='viridis')
plt.title('Entropy of input Image')
plt.axis('off')

print("Entropy of Colorized Image is: ", end="")
print(round(shannon_entropy(colorized[:,:,0]), 6))
entr_img2 = entropy(colorized[:,:,0], disk(10))
fig.add_subplot(2, 3, 6)
plt.imshow(entr_img2, cmap='viridis')
plt.title('Entropy of colorized Image')
plt.axis('off')

plt.show()

# cv2.imshow("difference", diff)
cv2.waitKey(0)
cv2.destroyAllWindows()