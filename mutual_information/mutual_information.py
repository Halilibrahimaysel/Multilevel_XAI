# -*- coding: utf-8 -*-


import pandas as pd
from utils import *
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
import math
import numpy as np
np.set_printoptions(suppress=True)


#load attributr list
location=r"."
attributes=pd.read_csv(location, sep=" ")
predicates=attributes["predicate"].tolist()


#load class list
classes = pd.read_csv(r".", sep=" ")
classes = classes["class"].to_list()


#load features extracted using ResNet101
feature_list=list()

for i in range(2048):
    feature_list.append("feature{}".format(i))
    
#2048 attributes extracted from ResNet101 for all 40k images
attributes = pd.read_csv(r".\AWA2_dataset\AwA2-features\Animals_with_Attributes2\Features\ResNet101\AwA2-features.txt", sep=" ", names=feature_list)
# attributes=attributes.drop("Unnamed: 0", axis=1)

#import label of classes of 40k images
labels=pd.read_csv(r".\AWA2_dataset\AwA2-features\Animals_with_Attributes2\Features\ResNet101\AwA2-labels.txt", sep=" ", names=["label"])

#convert labels to 0-based from 1-based
for i in range(len(labels)):
    labels.label[i] = labels.label[i]-1
    
#train-test split  
X_train, X_test, y_train, y_test = train_test_split(attributes, labels, test_size=0.3, random_state = 42, shuffle=True, stratify=labels)
del attributes

#load the model which is trained before (MLP+language model)
model = load_model(r".")


#create a model that gives 85 features as output
att_model = Model(inputs=model.input, outputs=(model.get_layer("dense").output))

#extract 85 feature responses
att = att_model.predict(X_test)

#binarize the att values based on a threshold
binarized = (att >= 2) * att
binarized[binarized >= 2] = 1


#create a dataframe (11197,86), columns are 85 attributes and class label
df = pd.DataFrame(binarized, columns = predicates)
df["labels"] = y_test.values



#information using test images
I_list = []

for i in range(len(predicates)):

    D=len(df)
    
    attribute = predicates[i]
    list_cond_entropy = []
    list_class_entropy = []
    
    for i in range(len(classes)):
        
        filter_by_class = df['labels']==i
        label = df[filter_by_class]
    
        P_class = len(label)/D
        H_class = P_class*math.log2(P_class)
        
        list_class_entropy.append(H_class)
    
        #for Xi=1
        filter_class_by_att = label[attribute]==1
        horn_label = label[filter_class_by_att]
        

        filter_by_att = df[attribute]==1
        horns = df[filter_by_att]
        

        first = len(horn_label)/D
        second = len(horns)/D
        third = (first/(second+1e-9))        
        forth = first*math.log2(third+1e-9)
        list_cond_entropy.append(forth)

        
        #for Xi=0
        absence_filter_class_by_att = label[attribute]==0
        absence_horn_label = label[absence_filter_class_by_att]
        
        absence_filter_by_att = df[attribute]==0
        absence_horns = df[absence_filter_by_att]
        
        absence_first = len(absence_horn_label)/D
        absence_second = len(absence_horns)/D
        absence_third = (absence_first/(absence_second+1e-9))
        absence_forth = absence_first*math.log2(absence_third+1e-9)
         
        list_cond_entropy.append(absence_forth)
    
    H_initial = -sum(list_class_entropy)
    H_class_att = -sum(list_cond_entropy)
    
    #mutual information
    I_class_att = H_initial-H_class_att
    I_list.append(I_class_att)
    


list_original = I_list
list_original = np.array(list_original)

final_df = pd.DataFrame(data=list_original, index=predicates)




#information in 50*85 matrix (without images)

data,classes,features=load_AwA2()   #load the binary matrix
data['labels'] = labelencoder.fit_transform(range(50))
I_list = []


df=data

for i in range(len(predicates)):

    D=len(df)
    
    attribute = predicates[i]
    list_cond_entropy = []
    list_class_entropy = []
    
    for i in range(len(classes)):
        
        filter_by_class = df['labels']==i
        label = df[filter_by_class]
    
        P_class = len(label)/D
        H_class = P_class*math.log2(P_class)
        
        list_class_entropy.append(H_class)
    
        #for Xi=1
        filter_class_by_att = label[attribute]==1
        horn_label = label[filter_class_by_att]
        

        filter_by_att = df[attribute]==1
        horns = df[filter_by_att]
        

        first = len(horn_label)/D
        second = len(horns)/D
        third = (first/(second+1e-9))        
        forth = first*math.log2(third+1e-9)
        list_cond_entropy.append(forth)

        
        #for Xi=0
        absence_filter_class_by_att = label[attribute]==0
        absence_horn_label = label[absence_filter_class_by_att]
        
        absence_filter_by_att = df[attribute]==0
        absence_horns = df[absence_filter_by_att]
        
        absence_first = len(absence_horn_label)/D
        absence_second = len(absence_horns)/D
        absence_third = (absence_first/(absence_second+1e-9))
        absence_forth = absence_first*math.log2(absence_third+1e-9)
         
        list_cond_entropy.append(absence_forth)
    
    H_initial = -sum(list_class_entropy)
    H_class_att = -sum(list_cond_entropy)
    
    #mutual information
    I_class_att = H_initial-H_class_att
    I_list.append(I_class_att)
    


list_original = I_list
list_original = np.array(list_original)

final_org_df = pd.DataFrame(data=list_original, index=predicates)









