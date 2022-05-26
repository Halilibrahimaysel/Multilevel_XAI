# -*- coding: utf-8 -*-


from tensorflow.keras.models import load_model
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.layers import *
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras import layers
from tensorflow.keras.applications.resnet import ResNet101
from tensorflow.keras.optimizers import Adam
np.set_printoptions(suppress=True)
import matplotlib.cm as cm
from tensorflow import keras
from keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing import image
import tensorflow
import cv2
from utils import *
from PIL import Image
import scipy
np.set_printoptions(suppress=True)


#load attribute list
location=r"."
attributes=pd.read_csv(location, sep=" ")
predicates=attributes["predicate"].tolist()

#load class list
classes = pd.read_csv(r".", sep=" ")
classes=classes["class"].to_list()

#read an image
img = load_img(r".", target_size=(256,256))
img = np.array(img)

#plot img
fig, ax = plt.subplots()
ax.imshow(img)
ax.axis("off")
plt.show()

img_tensor = np.expand_dims(img, axis=0)
preprocessed_img = preprocess_input(img_tensor)


# #create a model to get 8*8*2048 feature maps of the above image
baseModel = ResNet101(weights="imagenet", include_top=False)
model = Model(inputs=baseModel.input, outputs=baseModel.get_layer("conv5_block3_out").output)

# # get 8*8*2048 features of the img
features_img = model.predict(preprocessed_img)

final_model = load_model(r".") 
#get the class of img_test_dims
prediction = final_model.predict(features_img)
print(classes[np.argmax(prediction)], np.max(prediction))

#create multiple model to see the activation at different layer (last CNN, attributes layer etc.)
    #model1 -- pre_activation to X-CNN layer
    #model2 -- after_activation to X-CNN layer
    #model3 -- 85 attributes model   
multiple_model = Model(inputs=final_model.input, outputs=(final_model.get_layer("conv2d").output, final_model.get_layer("activation").output, final_model.get_layer("batch_normalization").output))

# #feed the 8*8*85 features and get the pre and after sigmoid feature maps
pre_act_map, after_act_map, attribute_values = multiple_model.predict(features_img)
pre_act_map = np.squeeze(pre_act_map)
after_act_map = np.squeeze(after_act_map)
attribute_values = np.squeeze(attribute_values)

#sort attributes to see the importance order
attributes_sorted = np.argsort(attribute_values)[::-1]


### get the highest and lowest gradients w.r.t language model input using utils.py
gradients_descending, grads = Best_gradients(final_model, "batch_normalization", features_img)

#print the most important attributes and their indexes
for attribute in (gradients_descending):
    print("feature:",predicates[attribute], "index:{}".format(attribute), "gradient:",tf.get_static_value(grads[0][attribute]))


# #extract nth map and upsample it to superimpose original image
n=30
nth_map = after_act_map[:,:,30]

#upsample 8*8 to input image size
h = int(img.shape[0]/nth_map.shape[0])
w = int(img.shape[1]/nth_map.shape[1])
upsampled_map = scipy.ndimage.zoom(nth_map,(h,w), order=1)

#plot heatmap
fig, ax = plt.subplots()
ax.imshow(img)
ax.imshow(upsampled_map, cmap="jet", alpha=0.5)
ax.axis("off")
plt.show()

# #check the original attribute-class matrix to see if the outputted attribute by our model is actually important
data,classes,features=load_AwA2()

# data = df.astype(float)
data['labels'] = labelencoder.fit_transform(range(50))
data.index = classes
#normalise the values to 0-1
samples, labels_samples=random_sampling(data, 50, 0)
samples.index = classes
data.loc["walrus"]["fish"]



























































