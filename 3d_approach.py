!pip install patchify

#the classification-model-3D and efficientnet-3D are 2 dependencies for segmentation-models-3D library.
!pip install classification-models-3D
!pip install efficientnet-3D
!pip install segmentation-models-3D




import tensorflow as tf
import keras
#for using 3D unet
import segmentation_models_3D as sm
#for reading 3D image specially in tif extension
from skimage import io
#for dividing the full 3D image and reverse the process
from patchify import patchify, unpatchify
import numpy as np
from matplotlib import pyplot as plt
from keras import backend as K
#for one-hot encoding
from keras.utils import to_categorical
#For splitting the data to train val and test splits.
from sklearn.model_selection import train_test_split


image = io.imread('image/path/train_images_256_256_256.tif')
#using step argument = 64 mean that our patches have no overlaping voxels
img_patches = patchify(image, (64, 64, 64), step=64)   


mask = io.imread('image/path/train_masks_256_256_256.tif')
mask_patches = patchify(mask, (64, 64, 64), step=64)  

input_img = np.reshape(img_patches, (-1, img_patches.shape[3], img_patches.shape[4], img_patches.shape[5]))
input_mask = np.reshape(mask_patches, (-1, mask_patches.shape[3], mask_patches.shape[4], mask_patches.shape[5]))

print(input_img.shape)  # n_patches, x, y, z
# (64,64,64,64)


train_img = np.stack((input_img,)*3, axis=-1)
#only one channels for the mask
train_mask = np.expand_dims(input_mask, axis=4)


def soft_dice_coefficient(y_true, y_pred):
    smoothing_factor = 1
    flat_y_true = K.flatten(y_true)
    flat_y_pred = K.flatten(y_pred)
    return (2. * K.sum(flat_y_true * flat_y_pred) + smoothing_factor) / (K.sum(K.square(flat_y_true)) + K.sum(K.square(flat_y_pred)) + smoothing_factor)

def soft_dice_loss(y_true, y_pred):
    return 1 - soft_dice_coefficient(y_true, y_pred)

encoder_weights = 'imagenet'
BACKBONE = 'vgg16'  
activation = 'sigmoid'
patch_size = 64
channels=3


preprocess_input = sm.get_preprocessing(BACKBONE)

#Preprocess input data 
X_train_prep = preprocess_input(train_img)


#Define the model. Here we use 3D unet architectures from the library with vgg16 encoder backbone.
model = sm.Unet(BACKBONE, 
                input_shape=(patch_size, patch_size, patch_size, channels), 
                encoder_weights=encoder_weights,
                activation=activation)

model.compile(optimizer = "Adam", loss=soft_dice_loss, metrics=soft_dice_coefficient)


#Fit the model
history=model.fit(X_train_prep, 
          train_mask,
          batch_size=16, 
          epochs=100)
   

#Break the large image into patches of same size as the training images (patches)
image = io.imread('/mypath/51x512x512.tif')
image_patches = patchify(image, (64, 64, 64), step=64)  #Step=64 for 512 pexel means no overlap




# Predict each 3D patch   
predicted_patches = []
for i in range(image_patches.shape[0]):
  for j in range(image_patches.shape[1]):
    for k in range(image_patches.shape[2]):
      #print(i,j,k)
      single_patch = image_patches[i,j,k, :,:,:]
      single_patch_3ch = np.stack((single_patch,)*3, axis=-1)
      single_patch_3ch_input = preprocess_input(np.expand_dims(single_patch_3ch, axis=0))
      single_patch_prediction = model.predict(single_patch_3ch_input)
      binary_mask = np.where(single_patch_prediction >= 0.5, 1, 0)

      predicted_patches.append(binary_mask)



#Convert list to numpy array
predicted_patches = np.array(predicted_patches)

#Reshape to the shape we had after patchifying
predicted_patches_reshaped = np.reshape(predicted_patches, 
                                       (image_patches.shape[0], image_patches.shape[1], image_patches.shape[2],patches.shape[3], patches.shape[4], patches.shape[5]) )
    
#Repach individual patches into the orginal volume shape
reconstructed_image = unpatchify(predicted_patches_reshaped,image.shape)


# Define the voxel spacing in millimeters
voxel_spacing = 1.0

# Compute the volume of the segmented pancreas in millimeters cubed
pancreas_volume = np.sum(reconstructed_image) * (voxel_spacing ** 3)
     

