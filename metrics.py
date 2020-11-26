from skimage import data
from skimage.color import rgb2gray
import numpy as np

import sklearn
from sklearn import metrics

import numpy as np
from skimage import data, img_as_float
from skimage.measure import compare_ssim as ssim
import math

from PIL import Image

#this code is adjustable to the other mappings by changing the directory of the synthesized and real images

Path_Images = "./images/"

ZEROES = 5

def image_number(n):
    number = str(n)

    while len(number) < ZEROES:
        number = "0" + number
    
    return number

def generate_images_modern_T1T2(model, test_input, tar, epoch, i):
    intel = i
    epoch_1 = epoch
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15,15))

    prediction_map = prediction[0]
    tar_map = tar[0]
    test_input_map = test_input[0]

    prediction_gray = rgb2gray(prediction_map)
    tar_gray = rgb2gray(tar_map)

    error = np.abs(prediction_gray[:,:]-tar_gray[:,:])

    display_list = [test_input[0], tar[0], prediction[0], error]
    title = ['Input Image', 'Ground Truth', 'Predicted Image', "Error Image"]
    
    #### mean squared error ####
    loss1 = sklearn.metrics.mean_squared_error(tar_gray, prediction_gray)
    loss1 = round(loss1, 3)

    ### structural similarity index #####
    loss2 = ssim(tar_gray, prediction_gray)
    loss2 = round(loss2, 3)
    
    ##### peak signal to noise ratio (PSNR) ########
    
    loss3 = 20*math.log(4095/loss1, 10)
    loss3 = round(loss3, 3)
    
    ##### mean absolute error #######
    loss4 = sklearn.metrics.mean_absolute_error(tar_gray, prediction_gray)
    loss4 = round(loss4, 3)
    
    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.title(title[i], fontsize = 16)
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
        
    plt.suptitle("Epoch: "+str(epoch_1)+"         MSE: "+str(loss1)+"      SSI "+str(loss2)+"    PSNR: "+str(loss3)+ "     MAE: "+str(loss4), x=0.5, y = 0.95, verticalalignment = "center", fontsize=22)
    #plt.show()
    
    plt.savefig(os.path.join(Path_Images, "BikeT1T2"+image_number(epoch_1)+"_"+image_number(intel)+".jpg"))
    
def generate_images_modern_T2T1(model, test_input, tar, epoch, i):
    intel = i
    epoch_1 = epoch
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15,15))

    prediction_map = prediction[0]
    tar_map = tar[0]
    test_input_map = test_input[0]

    prediction_gray = rgb2gray(prediction_map)
    tar_gray = rgb2gray(tar_map)

    error = np.abs(prediction_gray[:,:]-tar_gray[:,:])

    display_list = [test_input[0], tar[0], prediction[0], error]
    title = ['Input Image', 'Ground Truth', 'Predicted Image', "Error Image"]
    
    #### mean squared error ####
    loss1 = sklearn.metrics.mean_squared_error(tar_gray, prediction_gray)
    loss1 = round(loss1, 3)

    ### structural similarity index #####
    loss2 = ssim(tar_gray, prediction_gray)
    loss2 = round(loss2, 3)
    
    ##### peak signal to noise ratio (PSNR) ########
    
    loss3 = 20*math.log(4095/loss1, 10)
    loss3 = round(loss3, 3)
    
    ##### mean absolute error #######
    loss4 = sklearn.metrics.mean_absolute_error(tar_gray, prediction_gray)
    loss4 = round(loss4, 3)
    
    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.title(title[i], fontsize = 16)
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
        
    plt.suptitle("Epoch: "+str(epoch_1)+"         MSE: "+str(loss1)+"      SSI "+str(loss2)+"    PSNR: "+str(loss3)+ "     MAE: "+str(loss4), x=0.5, y = 0.95, verticalalignment = "center", fontsize=22)
    #plt.show()
    
    plt.savefig(os.path.join(Path_Images, "BikeT2T1"+image_number(epoch_1)+"_"+image_number(intel)+".jpg"))
