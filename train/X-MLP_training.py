# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 12:42:56 2022

@author: HALIL IBRAHIM
"""

from tensorflow.keras.models import load_model
import pandas as pd
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

feature_list=list()

for i in range(2048):
    feature_list.append("feature{}".format(i))
    
#2048 attributes extracted from ResNet101 for all 40k images
attributes = pd.read_csv(r"C:\Users\HALIL IBRAHIM\OneDrive - University of Southampton\New_language_based_idea\zero_shot_extensions\AWA2_dataset\AwA2-features\Animals_with_Attributes2\Features\ResNet101\AwA2-features.txt", sep=" ", names=feature_list)

#import label of classes of 40k images
labels=pd.read_csv(r"C:\Users\HALIL IBRAHIM\OneDrive - University of Southampton\New_language_based_idea\zero_shot_extensions\AWA2_dataset\AwA2-features\Animals_with_Attributes2\Features\ResNet101\AwA2-labels.txt", sep=" ", names=["label"])

#convert labels to 0-based from 1-based
for i in range(len(labels)):
    labels.label[i] = labels.label[i]-1

#one hot encode the labels
labels = to_categorical(labels, dtype ="uint8")

#class names in the same order of data file (not alphabetically)
classes = pd.read_csv(r"C:\Users\HALIL IBRAHIM\OneDrive - University of Southampton\New_language_based_idea\zero_shot_extensions\AWA2_dataset\AwA2-data\Animals_with_Attributes2\classes_nonumber.txt", sep=" ")
classes=classes["class"].to_list()

#train-test split
X_train, X_test, y_train, y_test = train_test_split(attributes, labels, test_size=0.3, random_state = 42, shuffle=True, stratify=labels)
del attributes

#load pre-trained language model
language_model = load_model(r'C:\Users\HALIL IBRAHIM\OneDrive - University of Southampton\New_language_based_idea\zero_shot_extensions\AWA2_dataset\My_code_for_AwA2\language_pre_training\model_8.h5', compile=False)

#freeze the layers of the language model
for layer in language_model.layers:
    layer.trainable=False

# Define  a functional model where extra layers added in front our trained language model 
# to match the 2048 attributes with 85 attributes
inputs=Input(shape=(2048,))

x=Dense(256, activation="relu", activity_regularizer=regularizers.l2(1e-5), name="layer1")(inputs)
x=BatchNormalization(axis=1)(x)
x=Dense(256, activation="relu", activity_regularizer=regularizers.l2(1e-5), name="layer2")(x)
x=BatchNormalization(axis=1)(x)
x=Dropout(0.2)(x)
x=Dense(128, activation="relu", activity_regularizer=regularizers.l2(1e-5), name="layer3")(x)
x=BatchNormalization(axis=1)(x)
x=Dropout(0.2)(x)
x=Dense(85, activation="relu")(x)
x=language_model(x)

model1=Model(inputs=inputs, outputs=x)
model1.summary()


#compile the model
model1.compile(optimizer="Adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])


#add callbacks
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
path=r"C:\Users\HALIL IBRAHIM\OneDrive - University of Southampton\New_language_based_idea\zero_shot_extensions\AWA2_dataset\My_code_for_AwA2\X-MLP\checkpoints_relu\weights_improvement-{epoch:02d}--{val_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(path, monitor='val_accuracy', verbose=2, save_best_only=True, mode='max')
early_stop = EarlyStopping(monitor='val_loss', patience=20, verbose=2) 

callback_list=[checkpoint, early_stop]


#fit and train the model
history=model1.fit(X_train, y_train, batch_size=32, epochs=100, 
                  verbose=2, validation_split=0.2,
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
y_pred_test = model1.predict(X_test)

y_pred_test_max = np.argmax(y_pred_test, axis=1)
y_test_max = np.argmax(y_test, axis=1)

print("Accuracy on test set :",accuracy_score(y_test_max, y_pred_test_max))










