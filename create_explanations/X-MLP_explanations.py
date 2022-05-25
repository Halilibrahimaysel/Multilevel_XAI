# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 11:25:56 2022

@author: HALIL IBRAHIM
"""

from tensorflow.keras.applications.resnet import ResNet101
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import preprocess_input
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img,img_to_array
import matplotlib.pyplot as plt
import tensorflow as tf
from utils import *
np.set_printoptions(suppress=True)


#load attribute list
location=r"C:\Users\HALIL IBRAHIM\OneDrive - University of Southampton\New_language_based_idea\zero_shot_extensions\AWA2_dataset\AwA2-data\Animals_with_Attributes2\predicates_nonumber.txt"
attributes=pd.read_csv(location, sep=" ")
predicates=attributes["predicate"].tolist()


#load class list
classes = pd.read_csv(r"C:\Users\HALIL IBRAHIM\OneDrive - University of Southampton\New_language_based_idea\zero_shot_extensions\AWA2_dataset\AwA2-data\Animals_with_Attributes2\classes_nonumber.txt", sep=" ")
classes=classes["class"].to_list()

#read an image
img = load_img(r"C:\Users\HALIL IBRAHIM\Desktop\Desktop\elephant.jpg", target_size=(256,256))
img = np.array(img)

#plot img
fig, ax = plt.subplots()
ax.imshow(img)
ax.axis("off")
plt.show()

img_tensor = np.expand_dims(img, axis=0)
preprocessed_img = preprocess_input(img_tensor)

# #create a model to get 8*8*2048 feature maps of the above image
baseModel = ResNet101(weights="imagenet", include_top=False,  input_shape=(256,256,3))

headModel = baseModel.output
headModel = layers.GlobalAveragePooling2D()(headModel)

headModel = Model(inputs=baseModel.input, outputs=headModel)


# # get 2048 features of first img
features_first_img = headModel.predict(preprocessed_img)

#load the model which is trained before (MLP+language model)
model=load_model(r"C:\Users\HALIL IBRAHIM\OneDrive - University of Southampton\New_language_based_idea\zero_shot_extensions\AWA2_dataset\My_code_for_AwA2\X-MLP\checkpoints_relu\weights_improvement-13--0.91.hdf5")


#get the class of img_test_dims
prediction = model.predict(features_first_img)
print(classes[np.argmax(prediction)], np.max(prediction))


# ######get the highest and lowest gradients using utils
gradients_descending, grads = Best_gradients(sigmoid_model, "dense", features_first_img)

#print the most important attributes and their indexes
for attribute in (gradients_descending):
    print("feature:",predicates[attribute], "index:{}".format(attribute), "gradient:",tf.get_static_value(grads[0][attribute]))


#investigate whats happening
model_2 = Model(inputs=model.inputs, outputs=model.get_layer("dense").output)
b=np.squeeze(model_2.predict(features_first_img))

#load AwA2 dataset with attributes using utils.py
data,classes,features=load_AwA2()

samples, labels_samples = random_sampling(data, 50, 0)

for i in range(50):
    
    a=samples.iloc[i]
    fark = a-b
    sum_fark=fark.sum()
    print(sum_fark)
