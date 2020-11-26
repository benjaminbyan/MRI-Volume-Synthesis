import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("fivethirtyeight")

from training import evaluate_model
from data_loader import normalize
from training import model 

ckpt.restore("checkpoints/Train/ckpt-11.index")

x_dimension = 256
y_dimension = 256

height = 256
width = 256

def special_loader(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)

    w = tf.shape(image)[1]

    w = w // 2
    input_image = image[:, :w, :]
    real_image = image[:, w:, :]

    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)
    
    input_image, real_image = resize(input_image, real_image,
                                    x_dimension, y_dimension)
    (input_image, real_image) = (tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR),
                              real_image = tf.image.resize(real_image, [height, width],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR))
    (input_image, real_image) = normalize(input_image,real_image)
    return input_image, real_image

Path_Input = "./Testing/"
Path_Images = "./SyntheticCE/"

index=0
i=0
sum = 0
intermed = []

  #takes the average
  #sum = float(sum/155)

  #print(str(sum))

def generate_images_modern_T1(model, test_input, tar, intel):
    intel = i
    #epoch_1 = epoch + 40
    prediction = model(test_input, training=True)
    #plt.figure(figsize=(15,15))

    prediction_map = prediction[0]
    tar_map = tar[0]
    test_input_map = test_input[0]

    plt.figure()
    plt.imshow(prediction[0]* 0.5 + 0.5) 
    
    #for i in range(4):
     #   plt.subplot(2, 2, i+1)
      #  plt.title(title[i], fontsize = 16)
        # getting the pixel values between [0, 1] to plot it.
       # plt.imshow(display_list[i] * 0.5 + 0.5)
       # plt.axis('off')
 
    plt.savefig(os.path.join(Path_Images, image_number(intel)+".jpg"))

while (i<155):
   j = i + 1550
   inp, tar = special_loader(Path_Input + image_number(j)+".jpg")
   inp = tf.expand_dims(inp,0)
   tar = tf.expand_dims(tar,0)
   #generate_images_modern_T1T2(generator_T1T2, inp, tar, 69, i)

   generate_images_modern_T1(generator_T1T2, inp, tar, i)
   #intermed.append(evaluate_model(generator_T1T2, inp, tar)[index])
   i+=1
