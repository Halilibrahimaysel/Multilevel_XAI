# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 13:05:39 2022

@author: HALIL IBRAHIM
"""

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
import numpy as np
from tensorflow.keras.layers import Input, Dense, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.layers import *
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras import layers
from tensorflow.keras.applications.resnet import ResNet101
from tensorflow.keras.optimizers import Adam
np.set_printoptions(suppress=True)

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# 8*8*2048 attributes extracted from the last CNN layer of ResNet101 for all 40k images
features = np.load(r"D:\datasets\AwA2\last_CNN_features_new.npy")

# labels
labels_alpha = np.load(r"D:\datasets\AwA2\labels.npy")
labels_alpha = to_categorical(labels_alpha, dtype ="uint8")

# # #load language model
language_model = load_model(r'C:\Users\HALIL IBRAHIM\OneDrive - University of Southampton\New_language_based_idea\zero_shot_extensions\AWA2_dataset\My_code_for_AwA2\language_pre_training\model_8.h5', compile=False)

#freeze the layer of the language model
for layer in language_model.layers:
    layer.trainable=False
    
# #Add the trained resnet features(8*8*2048) in front of MLP+Language model
# # # Define  a functional model where extra layer(CNN) added between the trained language model and 
# # # to match the 2048 attributes with 85 attributes
inputs=Input(shape=(8,8,2048,))

#the shape of input is 8*8*2048 we need add a CNN layer to reduce it to 8*8*85
#then a GAP to make it 85 only

model = Conv2D(85,1,padding="valid")(inputs)
model = Activation("relu")(model)

model = GlobalAveragePooling2D()(model) 
model = BatchNormalization()(model)

model = Model(inputs=inputs, outputs=model)
model = language_model(model.output)

final_model = Model(inputs=inputs, outputs=model)



# #compile the model
final_model.compile(optimizer=Adam(learning_rate=1e-3),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# # # #train-test split
X_train, X_test, y_train, y_test = train_test_split(features, labels_alpha, test_size=0.3, random_state = 42, shuffle=True, stratify=labels_alpha)
del features

# #add callbacks
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
path=r"C:\Users\HALIL IBRAHIM\OneDrive - University of Southampton\New_language_based_idea\zero_shot_extensions\AWA2_dataset\My_code_for_AwA2\X-CNN\checkpoints_relu\weights_improvement-{epoch:02d}--{val_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(path, monitor='val_accuracy', verbose=2, save_best_only=True, mode='max')
early_stop = EarlyStopping(monitor='val_loss', patience=20, verbose=2) 

callback_list=[checkpoint, early_stop]

# #fit and train the model
history=final_model.fit(X_train, y_train, batch_size=32, epochs=50, 
                  verbose=1, validation_split=0.2,
                    callbacks=callback_list
                  )

#check the accuracy-loss curves
history_df = pd.DataFrame(history.history)

history_df.loc[:, ['loss', 'val_loss']].plot()
history_df.loc[:, ['accuracy', 'val_accuracy']].plot()

print(("Best Validation Loss: {:0.4f}" +\
      "\nBest Validation Accuracy: {:0.4f}")\
      .format(history_df['val_loss'].min(), 
              history_df['val_accuracy'].max()))
    

# Evaluation on test set
y_pred_test = final_model.predict(X_test)

y_pred_test_max = np.argmax(y_pred_test, axis=1)
y_test_max = np.argmax(y_test, axis=1)

print("Accuracy on test set :",accuracy_score(y_test_max, y_pred_test_max))





    
    
    
    
    
    
    
    
    
    