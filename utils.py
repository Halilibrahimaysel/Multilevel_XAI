# -*- coding: utf-8 -*-


import pandas as pd
import random
from sklearn import preprocessing
min_max = preprocessing.MinMaxScaler()
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import cv2
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
labelencoder = LabelEncoder()
scaler = StandardScaler()




# create a number of samples and manipulate them 
# choose n random column and change their values
# we will use this sample to test the model's robustness
# takes the dataframe, the length of the dataset, the # of column to change
# return the samples as a dataframe

def random_sampling(df, n_samples, n_columns):
    
    #n_sample is the number of classes in the dataset
    
    #set variables
    samples=df.head(n_samples)
    X_sample=samples.drop("labels", axis=1)
    y_sample=samples.labels
    num_of_features=len(df.columns)

    #normalise the data (This is needed when continuos data is used)
    X_sample1 = X_sample.copy()
    X_sample1 = scaler.fit_transform(X_sample1.values)
    # X_sample1 = min_max.fit_transform(X_sample1.values)
    X_sample1 = pd.DataFrame(X_sample1, columns=(X_sample.columns))
    
    #randomise the data   
    #takes each row (datapoint) and randomly chooses n_columns of it and manipulates
    #note that for every single sample columns are chosen randomly (e.g. for the 1st sample column 1,5,7 may change while for 2nd sample column 3,5,8)
    for sample in range(n_samples):
        
        current_sample=X_sample1.iloc[sample]
    
        for column in range(n_columns):
            new_rand=random.sample(range(num_of_features-1),1)
            # print(new_rand)
            # print(current_sample[new_rand])
            
            if current_sample[new_rand][0]<=0:
                
                current_sample[new_rand]=1.5
              
            else:
                
                current_sample[new_rand]=-0.5
                
            X_sample1.iloc[sample]=current_sample

    return X_sample1, y_sample




#function for AwA2 loading
def load_AwA2():

    location=r".\AWA2_dataset\AwA2-data\Animals_with_Attributes2"
    attributes=r"."
    classes=r"."
    assignments=r"."
    
    attributes=pd.read_csv(location+attributes, sep=" ")
    predicates=attributes["predicate"].tolist()
    
    classes=pd.read_csv(location+classes, sep=" ")
    classes1=classes["class"].tolist()
    
    data=pd.read_csv(location+assignments, sep="  ", names=predicates)
    data["labels"]=classes1

    return data, classes1, predicates

#function for AwA2 loading
def load_CUB():

    location=r".\CUB"
    attributes=r"."
    classes=r"."
    assignments=r"."
    
    attributes=pd.read_csv(location+attributes, sep=" ")
    predicates=attributes["predicate"].tolist()
    
    classes=pd.read_csv(location+classes, sep=" ")
    classes1=classes["class"].tolist()
    
    data=pd.read_csv(location+assignments, sep=" ", names=predicates)
    data["labels"]=classes1

    return data, classes1, predicates





#Gradients
def Best_gradients(model, layer, img):
    # First, we create a model that maps the input image to the activations
    # of the layer that is input to the language model as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer).output, model.output]
    )
    
    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the layer just before the language model
    with tf.GradientTape() as tape:
        layer, preds = grad_model(img)
        pred_index = tf.argmax(preds[0])
        #for a specific index to choose (lets say I want to see the chimpanzee's gradients)
        # currently chooses the highest prediction to get gradients
        # pred_index=1
        class_channel = preds[:, pred_index]
        
    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the layer that is input to the language model
    grads = tape.gradient(class_channel, layer)
    
    # #show the 10 highest gradient for the investigated image
    gradients_descending=np.argsort(grads)[0][::-1]
    gradients_descending=gradients_descending.tolist()

    
    return gradients_descending, grads






























#shuffle attributes
def shuffle_data(data, mode):
    
    # y_sample = data.labels
    # data=data.drop("labels", axis=1)
    
    if mode == "column":
        shuffled_data = data.copy()
        for i in data.columns:
            shuffled_data[i] = shuffle(data[i]).values
  
    elif mode == "row":
        shuffled_data = data.copy()
        for i in data.index: 
            shuffled_data.iloc[i] = shuffle(data.iloc[i]).values
        
    elif mode == "complete":
        shuffled_data = data.copy()

        for i in data.columns:
            shuffled_data[i] = shuffle(data[i]).values 
            
        for j in shuffled_data.index: 
            shuffled_data.iloc[j] = shuffle(shuffled_data.iloc[j]).values

           
    else:
        print("please choose one of the modes in the list: [column,row,entire]")
        
    
    # shuffled_data["labels"]= y_sample
    
    return shuffled_data

        
        



#inputs feature_map(8*8) and img_test (256,256,3 image array)
#outputs heatmapped image and superimposed image
def draw_heatmap(feature_map, img_test):
    # Resize heatmap to the input size
    heatmap = np.copy(feature_map)
    
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(heatmap)
    
    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")
    
    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    
    
    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img_test.shape[1], img_test.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)
    
    # Superimpose the heatmap on original image
    heatmap_img = jet_heatmap * 0.5 + img_test


    #and black out not important pixels for better visualization
    heatmap_1 = cv2.resize(heatmap,(256, 256),interpolation=cv2.INTER_CUBIC)
    heatmap_1 = np.expand_dims(heatmap_1, axis=-1)
    superimposed_img =  heatmap_1/255 * img_test/255
    superimposed_img = (superimposed_img)
    
    return heatmap_img, superimposed_img
        








#a function to replace rows with the columns and vice versa (this is done to make the columns features)
#takes a dataframe, list of features and list of classes and returns the converted version

def column_to_row(data, classes, features):
    
    df=pd.DataFrame(index=classes, columns=features)
    for column in range(len(classes)):
        for row in range(len(features)):        
            df.iloc[column][row]=data[classes[column]][row]
    
    df = df.astype(float)
    df['labels'] = labelencoder.fit_transform(df.index)
    
    return df







