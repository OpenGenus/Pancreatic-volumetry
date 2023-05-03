

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, Dropout, Lambda
from keras.optimizers import Adam
from keras.layers import Activation, MaxPool2D, Concatenate


def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = Activation("relu")(x)

    return x

#Encoder block: Conv block followed by maxpooling


def encoder_block(input, num_filters):
    x = conv_block(input, num_filters)
    p = MaxPool2D((2, 2))(x)
    return x, p  

#Decoder block
#skip features gets input from encoder for concatenation

def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

#Build Unet using the blocks
def build_unet(input_shape):
    inputs = Input(input_shape)

    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    b1 = conv_block(p4, 1024) #Bridge

    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)  #Binary 

    model = Model(inputs, outputs, name="U-Net")
    return model



import numpy as np
import tensorflow as tf

# Load the 3D medical image data this time is .npy extension of shape(256,256,32) also the mask
train_images = np.load('image_data.npy')
train_labels = np.load('segmentation_data.npy')

# Define the slice size
slice_size = 256

# Define the 2D U-Net model
my_model = build_unet((256, 256, 1))

# Compile the model
my_model.compile(optimizer="Adam", loss=soft_dice_loss, metrics=soft_dice_coefficient)


# Train the model slice-by-slice
for i in range(train_images.shape[0]): # patch size
    for j in range(train_images.shape[2]): # depth dimension
        # Extract the current slice from the 3D image and corresponding segmentation label
        slice_image = train_images[i, :, :, j]
        slice_label = train_labels[i, :, :, j]

        # Reshape the slice image and label to (1, slice_size, slice_size, 1)
        slice_image = np.expand_dims(slice_image, axis=0)
        slice_image = np.expand_dims(slice_image, axis=-1)
        slice_label = np.expand_dims(slice_label, axis=0)
        slice_label = np.expand_dims(slice_label, axis=-1)

        # Train the 2D U-Net on the current slice and label
        my_model.train_on_batch(slice_image, slice_label)






# Loop through each slice in the 3D image
slices = []
for i in range(train_images.shape[2]):
    # Extract the current slice from the 3D image
    slice_data = train_images[:, :, i]

    # Resize the slice to the desired size for the 2D U-Net model
    resized_slice = tf.image.resize(slice_data, [slice_size, slice_size])

    # Add the resized slice to the list of slices
    slices.append(np.expand_dims(resized_slice, axis=-1))

# Convert the list of slices to a NumPy array
slices = np.asarray(slices)

# Perform the segmentation on each slice using the 2D U-Net model
segmented_slices = []
for i in range(slices.shape[0]):
    # Perform the segmentation on the current slice
    segmented_slice = my_model.predict(np.expand_dims(slices[i], axis=0))

    # Add the segmented slice to the list of segmented slices
    segmented_slices.append(segmented_slice.squeeze())

# Convert the list of segmented slices to a NumPy array
segmented_slices = np.asarray(segmented_slices)

segmented_image = np.zeros_like(train_images)
for z in range(segmented_slices.shape[0]):
    # Extract the current segmented slice
    segmented_slice = segmented_slices[z, :, :]

    # Resize the segmented slice back to the original size of the 3D image
    resized_segmented_slice = tf.image.resize(segmented_slice, [train_images.shape[0], train_images.shape[1]])

    # Add the resized segmented slice to the segmented image
    segmented_image[:, :, z] = resized_segmented_slice
    
segmented_image = np.where(segmented_image>=0.5,1,0)






# Define the voxel spacing in millimeters
voxel_spacing = 1.0

# Compute the volume of the segmented pancreas in millimeters cubed
pancreas_volume = np.sum(segmented_image) * (voxel_spacing ** 3)
     
