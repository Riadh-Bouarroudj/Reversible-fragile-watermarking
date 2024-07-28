import cv2
import numpy as np 
import imageio.v2 as imageio
import os


current_directory = os.path.dirname(__file__)
img_path = os.path.join(current_directory, 'Images/Baboon.tiff')
wat_path = os.path.join(current_directory, 'Images/Peppers.tiff')


bloc_size=3                   #Use a block size of 3 or 6
img_size=512

Color_image=True              #False mean that images will be converted to grayscale
Normalization=True            #Normalize the image to the range [0,254] instead of [0,255] to avoid overflow and underflow problems
self_embed=True               #Meaning that the watermark is generated from the cover image
Auth_encryption=True          #Encrypt the authentication watermark using Henon map
key = (1.5, 2.3)
embedding_type="DWT"          #The method used during watermark generation


#It's preferable to not modify this variables
BPP=1                             
wat_size=int(img_size/bloc_size/2)

# Load the cover image
img = imageio.imread(img_path)
if np.max(img)>255 or np.min(img)<0:
    raise ValueError("Please use 8-bit images with pixel values ranging between 0 and 255")
"""print(img[0][0:3])"""

original_img=cv2.resize(img,(img_size,img_size))
if Color_image==False and len((np.asarray(original_img)).shape)==3:
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

#Normalize the cover image to the range of [4,251] to avoid overfloaw and underflow problems
if Normalization==True:
    lis=[]
    for channel in range (len((np.asarray(original_img)).shape)):      
        if len((np.asarray(original_img)).shape)==3:
            img=original_img[:, :, channel]
        else:
            img=original_img
        for i in range (img_size):
            for j in range (img_size):
                if img[i][j]<1:
                    img[i][j]=1
                elif img[i][j]>254:
                    img[i][j]=254
        lis.append(img)
    if len((np.asarray(original_img)).shape)==3:
        original_img = np.stack([lis[0], lis[1], lis[2]], axis=2)
    else:
        original_img=lis[0]


# Load the watermark
watermark=imageio.imread(wat_path)
if np.max(watermark)>255 or np.min(watermark)<0:
    raise ValueError("Please use 8-bit images with pixel values ranging between 0 and 255")
if len((np.asarray(original_img)).shape)!=3:
    watermark = cv2.cvtColor(watermark, cv2.COLOR_BGR2GRAY)
org_watermark=cv2.resize(watermark,(wat_size,wat_size)) 
