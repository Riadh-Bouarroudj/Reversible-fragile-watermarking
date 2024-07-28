import numpy as np
import pywt
import math
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from config_variables import *
import pickle
import os
import warnings

#Henon map
def encrypt_image(image, key):
    encrypted_image = np.copy(image)
    height, width = image.shape
    
    # Initialize the Hanon map with the given key
    x, y = key[0], key[1]
    
    for i in range(height):
        for j in range(width):
            # Generate pseudo-random numbers using the Hanon map
            x = 1 - key[0] * x**2 +  y
            y = key[1] *x
            
            # Encrypt the pixel value using XOR operation
            encrypted_image[i, j] = image[i, j] ^ int(x * 255)
    
    return encrypted_image

def decrypt_image(encrypted_image, key):
    decrypted_image = np.copy(encrypted_image)
    height, width = encrypted_image.shape
    
    # Initialize the Hanon map with the given key
    x, y = key[0], key[1]
    
    for i in range(height):
        for j in range(width):
            # Generate pseudo-random numbers using the Hanon map
            x = 1 - key[0] * x**2 +  y
            y = key[1] *x
            
            # Decrypt the pixel value using XOR operation
            decrypted_image[i, j] = encrypted_image[i, j] ^ int(x * 255)
    
    return decrypted_image


def Image_metrics(img1, img2):
   if len((np.asarray(img1)).shape)==3:
      long=3
   else:
      long=1

   result1=[]
   result2=[]
   result3=[]
   for channel in range (long):
      if long==3:
         im1=img1[:, :, channel]
         im2=img2[:, :, channel]
      else:
         im1=img1
         im2=img2
      
      msee=0
      for i in range(len(im1)):
         for j in range(len(im1)):
            if im1[i][j] >= im2[i][j]:
               msee = msee+int(((im1[i][j] - im2[i][j]) ** 2))
            else:
               msee = msee+int(((im2[i][j] - im1[i][j]) ** 2))
      mse=msee/(len(im1)*len(im1))
      result1.append(mse)
      
      if mse!=0:     
         psnr = 20 * math.log10(255 / math.sqrt(mse))
      else:
         psnr=400
      result2.append(psnr)

      ssim_value = ssim(im1,im2,data_range=255.0) 
      result3.append(ssim_value)

   if long==3:
      mse=(result1[0]+result1[1]+result1[2])/3
      psnr=(result2[0]+result2[1]+result2[2])/3
      ssim_v=(result3[0]+result3[1]+result3[2])/3
   else:
      mse=result1[0]
      psnr=result2[0]
      ssim_v=result3[0]
   
   if psnr==400:
      psnr="Inf"
   return mse, psnr, ssim_v

def dec_to_bin(n):
    x=bin(n).replace("0b", "")
    while len(x)<8 :
      x="0"+x 
    return (x)
    
def bin_to_dec(n):
    return int(n, 2)

def watermark_to_digit(wat):
  #Transform wat to 1D array
  wat = np.array(wat)
  wat = wat.flatten()
  #Put all the watermark bits in one string
  watermark=""
  for i in range (len(wat)):
         bi=dec_to_bin(wat[i])   
         watermark=watermark+bi
  return(watermark)



def embedding_DWT_watermark(original_img,org_watermark) :
  cover=np.copy(original_img)
  if len((np.asarray(cover)).shape)==3:
     long=3
  else:
     long=1

  # Load the watermarks
  if self_embed==True:
      Auth_wat=self_embedding(cover)
  else:
      Auth_wat = org_watermark

  Auth_arr=[]
  max_subband=[]
  #Prepare the watermarks and transform them to long binary digits
  for channel in range(long):
    if long==3:
        Auth=Auth_wat[:, :, channel]
    else:
        Auth=Auth_wat
    if Auth_encryption==True:
        Auth = encrypt_image(Auth, key) 
    Auth = watermark_to_digit(Auth)
    Auth_arr.append(Auth)
  
  w_comp_arr=[]
  #Loop on the RGB channels of the cover image
  for channel in range (long): 
   if long==3:
      watermarked_img=cover[:, :, channel]
   else:
      watermarked_img=cover
   watermark=Auth_arr[channel]
   max_sub=[]

   # Apply Discrete wavelet transform to the cover image
   coeffs = pywt.dwt2(watermarked_img, 'haar')             
   LL, (LH, HL, HH) = coeffs
   subband=HH
   #Loop on the frequency subbands of the image
   a=0
   power=(int(bloc_size/3))
   #Round the coefficient values to 5 numbers after the decimal point to avoid problems caused by DWT-IDWT 
   subband=np.round(subband,5)
   #Loop on each subband  
   while a+bloc_size<=int(len(subband)): 
     b=0
     while b+bloc_size<=int(len(subband)):               
         v=0   
         #Loop on each bloc 
         j=0   
         while j<3:
            k=0
            while k<3:
               if  v*BPP==8: break        # Ensure that 8 bits are embedded in each block to provide tamper localization           
               max=-5000
               for m in range(int(bloc_size/3)):
                for n in range(int(bloc_size/3)):
                  if subband[a+power*j+m][b+power*k+n]>max:
                     max=subband[a+power*j+m][b+power*k+n]
               max_sub.append(max)      

               #Watermark bits embedding
               bits=str(watermark[0])
               watermark=watermark[1:len(watermark)]   
               find=False
               for m in range(2):
                for n in range(2):
                  if subband[a+power*j+m][b+power*k+n]==max and find==False:
                     find=True
                     if bits=='1':
                        subband[a+power*j+m][b+power*k+n]=subband[a+power*j+m][b+power*k+n]+1
               v=v+1
               k=k+1
            j=j+1
         b=b+bloc_size
     a=a+bloc_size
   max_subband.append(max_sub)
   HH=subband
   #Apply inverse DWT to the watermarked subbands
   watermarked_coeffs = LL, (LH, HL, HH)
   watermarked = pywt.idwt2(watermarked_coeffs, 'haar') 
   
   #Convert the watermarked channel to integer values
   if np.min(watermarked)<0:
      warnings.warn("Underflow encountered", UserWarning)
   if np.max(watermarked)>=255.5:
      warnings.warn("Overflow encountered", UserWarning)
      #print(np.max(watermarked))

   for i in range (img_size):
      for j in range (img_size):
         if watermarked[i][j]<0:
            watermarked[i][j]=0
         elif watermarked[i][j]>255:
            watermarked[i][j]=255
         else:
          p=watermarked[i][j] % 1  
          #if we put p to 0.6, the watermarked image will have an infinite PSNR (and the watermark can't be extracted) given that the rounding will eliminate the distortion introduced during embedding. For this reason we pur p to 0.4 to maintain the little distortion enabling the watermark to be extracted accurately
          if  p >0.4 :                                           
            watermarked_img[i][j]=int(watermarked[i][j])+1
          else :
            watermarked_img[i][j]=int(watermarked[i][j])  
   w_comp_arr.append(watermarked_img)

  if long==3:
      watermarked_img=np.stack([w_comp_arr[0], w_comp_arr[1], w_comp_arr[2]], axis=2)
  else:
      watermarked_img=w_comp_arr[0]
  
  with open(os.path.join(os.path.dirname(__file__), 'Images/Max_subband.pkl'), 'wb') as file:
    pickle.dump(max_subband, file)
  
  return(watermarked_img,max_subband) 
 

def extraction_DWT_watermark(imagex,max_subband):
    if len((np.asarray(imagex)).shape)==3:
       long=3
    else:
       long=1
    image=np.copy(imagex)

    FAuth_watermark=[]
    res_arr=[]
    #Loop on the watermarked image channels
    for channel in range (long): 
     if long==3:
         image=imagex[:, :, channel]
     else:
        image=imagex
     restored_img=np.copy(image)
     restored_img = np.array(restored_img.astype("uint8"))

     max_sub=max_subband[channel]
     Auth_watermark=[]
     z=0
     # Apply Discrete wavelet transform to the channel
     coeffs = pywt.dwt2(image, 'haar')             
     LL, (LH, HL, HH) = coeffs
     subband=HH

     #Round the coefficient values to 5 numbers after the decimal point to avoid problems caused by DWT-IDWT 
     subband=np.round(subband,5)
     #Loop on each frequency subband
     a=0
     power=(int(bloc_size/3))
     while a+bloc_size<=len(subband):       
        b=0
        while b+bloc_size<=len(subband):
            wat=""
            v=0
            #Loop on each block
            j=0
            while j<3:
               k=0
               while k<3:
                  #Stop if 8 bits are extracted from the current block or if all the watermark bits have been extracted
                  if v*BPP==8 or len(Auth_watermark)==wat_size*wat_size:  break     

                  max1=-5000
                  x1=0
                  y1=0
                  for m in range(int(bloc_size/3)):
                    for n in range(int(bloc_size/3)):
                     if subband[a+power*j+m][b+power*k+n]>max1:
                        max1=subband[a+power*j+m][b+power*k+n]   
                        x1=a+power*j+m
                        y1=b+power*k+n

                  if max1>max_sub[z]+0.1: 
                     bit='1'   
                     subband[x1][y1]=subband[x1][y1]-1
                  else: bit='0'   
                  wat=wat+bit
                  z=z+1
 
                  # if 8 bits have been extracted from the current block, append them to their corresponding subband
                  if len(wat)==8:          
                     wat=bin_to_dec(wat)  
                     Auth_watermark.append(wat)
                     wat=""
                  v=v+1  
                  k=k+1
               j=j+1      
            b=b+bloc_size
        a=a+bloc_size

     HH=subband
     #Reconstruct and decrypt the extracted watermarks 
     Auth_watermark=np.array(Auth_watermark)
     Auth_watermark=Auth_watermark.reshape(wat_size,wat_size)
     Auth_watermark = np.array(Auth_watermark.astype("uint8"))
     if Auth_encryption==True:
        Auth_watermark = decrypt_image(Auth_watermark, key)
     FAuth_watermark.append(Auth_watermark)

     #Apply inverse DWT to the watermarked subbands
     restored_coeffs = LL, (LH, HL, HH)
     restored = pywt.idwt2(restored_coeffs, 'haar') 
     #Convert the watermarked channel to integer values
     for i in range (img_size):
      for j in range (img_size):
         p=restored[i][j] % 1 
         if  p >0.6 :                                           
            restored_img[i][j]=int(restored[i][j])+1
         else :
            restored_img[i][j]=int(restored[i][j])   
     res_arr.append(restored_img)

    if long==3:
      Auth_watermark =  np.stack([FAuth_watermark[0], FAuth_watermark[1], FAuth_watermark[2]], axis=2)
      restored_img =np.stack([res_arr[0], res_arr[1], res_arr[2]], axis=2)
    else:
      Auth_watermark =  FAuth_watermark[0]
      restored_img =res_arr[0]
    return(Auth_watermark,restored_img)
   

def self_embedding(imagex):
   img=np.copy(imagex)
   if len((np.asarray(img)).shape)==3:
     long=3
   else:
     long=1
   ww_arr=[]
   for channel in range (long): 
      if long==3:
         image=img[:, :, channel]
      else:
         image=img
      #In case of 12bit or 16bit image, normalize the image to to an 8-bit image
      if np.max(np.abs(image))>256:
         if np.max(np.abs(image))<4096:
            maaax=4095
         else: 
            maaax=65535
         img_norm = (image/ maaax) * 255
         image=np.round(img_norm,0)   
         if np.min(image)<0:
            image = (image + 255) /2 
  
      if embedding_type=='DWT':
         coeffs = pywt.dwt2(image, 'haar')
         LL, (LH, HL, HH) = coeffs
         #Divide by 1100 instead of max during normalization to maintain accurate tamper localization against collage and content addition attacks
         LL = (LL /600) * 255
         image=LL

      down_size=int(len(image[0])/wat_size)
      watermark = np.array([[0 for j in range(wat_size)] for i in range(wat_size)], dtype='uint8')
      ii=0
      i=0 
      while i+down_size < len(image):
         j=0
         jj=0
         while j+down_size <len(image):
             s=0
             for k in range(down_size):
              for m in range(down_size):
                 s=s+image[i+k][j+m]
             sum=s/(down_size*down_size)
             watermark[ii][jj]=int(round(sum,0))
             jj=jj+1
             j=j+down_size
         i=i+down_size
         ii=ii+1
      ww_arr.append(watermark)

   if long==3:
      watermark =  np.stack([ww_arr[0], ww_arr[1], ww_arr[2]], axis=2)
   else:
      watermark =  ww_arr[0]
   return(watermark)

def Tamper_detection(org_watermar,ext_watermar): 
   #Given that the embedding is done bit-by-bit, val represents the number of different bits we tolerate between the binary reprentation of two pixels
   val=0  
   total=0
   if len((np.asarray(org_watermar)).shape)==3:
     long=3
   else:
     long=1

   t_arr=[]
   # 0 for altered pixels and 1 for unaltered ones
   tamper = np.array([[0 for j in range(wat_size)] for i in range(wat_size)], dtype='uint8')
   for channel in range (long): 
      if long==3:
         og_watermark=org_watermar[:, :, channel]
         ex_watermark=ext_watermar[:, :, channel]
      else:
         og_watermark=org_watermar
         ex_watermark=ext_watermar
      for i in range(wat_size):
         for j in range(wat_size):
            if og_watermark[i][j]>ex_watermark[i][j]:
               diff=og_watermark[i][j]-ex_watermark[i][j]
            else:
               diff=ex_watermark[i][j]-og_watermark[i][j]
            #We use this thereshold only when the authentication watermark is genrated from the cover image, otherwise no need to use a threshold of 3
            if (diff<3) and (self_embed==True):  
                  tamper[i][j]=1 
            else : 
                  pixel=dec_to_bin(int(og_watermark[i][j]))
                  pixel2=dec_to_bin(int(ex_watermark[i][j]))
                  sum=0
                  for k in range(len(pixel)):
                     if pixel[k]!=pixel2[k]:
                        sum=sum+1
                  if sum>val:
                     tamper[i][j]=0
                     total=total+sum
                  else:
                     tamper[i][j]=1
      t_arr.append(tamper)
   
   final_tamper = np.copy(tamper)
   if long==3:
      for i in range (wat_size):
         for j in range (wat_size):
            if t_arr[0][i][j]==1 and t_arr[1][i][j]==1 and t_arr[2][i][j]==1:
               final_tamper[i][j]=1
            else:
               final_tamper[i][j]=0
   else:
      final_tamper=t_arr[0]

   BER=total/(wat_size*wat_size*8)/long*100
   print("Bit error rate BER: ",BER,"%")
   return(final_tamper)

def Tamper_localization(tamper):
   #Perform a mojority vote between neighboords, if a pixel is unaltered by 4 or more of its neighboords are altered, the pixel is considered altered
   tamperx=np.copy(tamper)
   for i in range(wat_size):
      for j in range(wat_size):
         #Ensure that the pixel is not an edge pixel to perform majority vote
         if tamperx[i][j]==1 and i>0 and i<wat_size-1 and j>0 and j<wat_size-1:
            som=0
            ii=-1
            while ii<=1:
               jj=-1
               while jj<=1:
                  if tamperx[i+ii][j+jj]==0:
                     som=som+1
                  jj=jj+1  
               ii=ii+1
            if som>=4:
               tamper[i][j]=0

   #Can use dilatation and erosion operations, but the accuracy is not optimal
   #tamper = cv2.dilate(tamper, (3,3), iterations=1)
   #tamper = cv2.erode(tamper, (3,3), iterations=1)
   return(tamper)

def Display_watermarked_image(original_img,watermarked_img,restored_img):
   fig, (ax1, ax2,ax3) = plt.subplots(1, 3)
   ax1.imshow(original_img,cmap='gray')
   ax1.set_title('Original image')
   ax2.imshow(watermarked_img,cmap='gray')
   ax2.set_title('Watermarked image')
   ax3.imshow(restored_img,cmap='gray')
   ax3.set_title('Restored image')
   plt.show()
   
   mse_value, psnr_value, ssim_value = Image_metrics(original_img, watermarked_img)
   print("Watermarked image PSNR : ",psnr_value,"\t \t Watermarked image MSE : ",mse_value,"\t \t Watermarked image SSIM : ",ssim_value)

   mse_value, psnr_value, ssim_value = Image_metrics(original_img, restored_img)
   print("Restored image PSNR : ",psnr_value,"\t \t Restored image MSE : ",mse_value,"\t \t Restored image SSIM : ",ssim_value)

def Display_watermark(org_water,Auth_watermark):
   tamper=Tamper_detection(org_water,Auth_watermark)

   fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
   ax1.imshow(org_water,cmap='gray')
   ax1.set_title('Original watermark')
   ax2.imshow(Auth_watermark,cmap='gray')
   ax2.set_title('Extracted watermark')
   ax3.imshow(tamper, cmap='binary',vmin=0, vmax=1)
   ax3.set_title('Tampering')
   plt.show()
