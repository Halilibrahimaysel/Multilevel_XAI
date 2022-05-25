# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from utils import *
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model



#load AwA2 dataset with attributes using utils.py
data,classes,features=load_AwA2()

data['labels'] = labelencoder.fit_transform(range(50))
  
#standardized original attribute-class matrix used as test set
samples, labels_samples=random_sampling(data, 50, 0)


#create an upsampled dataset
X=pd.DataFrame(columns=data.columns[:-1])
y = pd.Series(dtype='object')

# upsample the data by calling the randomising
#100 is the upsampling rate, 50 is the number of classes and 8 is the number of attributes to be manipulated
for i in range(100):
    features, labels=random_sampling(data, 50, 8) 
    
    X=pd.concat([X, features],axis=0, ignore_index=True)
    y=pd.concat([y, labels], axis=0, ignore_index=True)
    

#one hot encode the labels
y = to_categorical(y, dtype ="uint8")

#Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42, shuffle=True, stratify=y)

# Create an MLP
inputs=layers.Input(shape=(85,))

x = layers.Dense(256,activation="relu", name="layer1")(inputs)
x = layers.Dense(256,activation="relu", name="layer2")(x)
x = layers.Dense(50, name="layer3")(x)
x = layers.Activation("softmax", name="activation")(x)

model = Model(inputs=inputs, outputs=x)

#compile the model
model.compile(optimizer="Adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

#fit and train the model
history=model.fit(X_train, y_train, batch_size=32, epochs=100, verbose=2, validation_split=0.2)

history = pd.DataFrame(history.history)

history.loc[:, ['loss', 'val_loss']].plot()
history.loc[:, ['accuracy', 'val_accuracy']].plot()

print(("Best Validation Loss: {:0.4f}" +\
      "\nBest Validation Accuracy: {:0.4f}")\
      .format(history['val_loss'].min(), 
              history['val_accuracy'].max()))
    
# Evaluation on test set
y_pred_test = model.predict(X_test)

y_pred_test_max = np.argmax(y_pred_test, axis=1)
y_test_max = np.argmax(y_test, axis=1)


#evaluation on original attribute-class matrix used as test set
sample_pred = model.predict(samples)
sample_pred_max = np.argmax(sample_pred, axis=1)


print("Accuracy on test-set is :", accuracy_score(y_test_max, y_pred_test_max))
print("Accuracy on original-test-set is :", accuracy_score(labels_samples, sample_pred_max))


#save the trained model
model.save(r".")


















