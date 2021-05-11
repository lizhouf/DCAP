# import modules
from emotion import *
from train import *
from knowledge import *
# all utilities
import json
import re
import pandas as pd
import numpy as np
import datetime
from nltk import word_tokenize
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import tensorflow.compat.v1 as tf
from random import sample
tf.disable_v2_behavior()
import re
import spacy
import collections
# select en_core_web_lg or en_core_web_md for different language packages
nlp = spacy.load("en_core_web_lg")
import re
import spacy
import collections
import random
import pickle

'''
Root path for different users
'''
# User_root="/Users/user_name/"
LF_root = "/Users/lizhoufan/"
HY_root = "/Users/angelayu/"
JY_root = "/Users/yinzh/"

This_root = LF_root # TO CHANGE

## The main part start from here:
# For OG, first label the number of hate speech using original dictionary and give emotion

# Analysis_Data
Analysis_Data_Path = This_root + 'Dropbox/Twitter_COVID_Group/Data/Analysis/' #+ 'Test/' # For Jerry Only (if not test then comment Test)
# Clean_Data
Clean_Data_Path = This_root + 'Dropbox/Twitter_COVID_Group/Data/Clean/' #+ 'Test/' # For Jerry Only
# Raw_Data
Raw_Data_Path = This_root + 'Dropbox/Twitter_COVID_Group/Data/Raw/' #+ 'Test/' # For Jerry Only

# File_Names.csv contains all directories required for the program.
Read_Dict = pd.read_csv(Analysis_Data_Path+'File_Names.csv')
# The csv path contains original tweets
old_data_path = Analysis_Data_Path + Read_Dict.iloc[0,1]
# The csv path contains new tweets yet to be input  
new_data_path = Raw_Data_Path + Read_Dict.iloc[1,1]
# The csv path contains the old hate word dictionary 
old_dict_path = Analysis_Data_Path + Read_Dict.iloc[2,1]

# The path indicates where to save the post-processed original tweets
old_data_save_path = Analysis_Data_Path + Read_Dict.iloc[0,2]
# The path indicates where to save the post-processed new tweets
new_data_save_path = Analysis_Data_Path + Read_Dict.iloc[1,2]
# The path indicates where to save the prediction of both original and new tweets
prediction_save_path = Analysis_Data_Path + Read_Dict.iloc[2,2]
# The path indicates where to save the new unigram dictionary
new_dict_save_path = Analysis_Data_Path + Read_Dict.iloc[3,2]
# The path indicates where to save the new bigram dictionary
new_two_gram_save_path = Analysis_Data_Path + Read_Dict.iloc[4,2]

# The paths indicate where to save the sub=models and meta-model
models_save_path = []
model1_save_path = Analysis_Data_Path + Read_Dict.iloc[5,2]
model2_save_path = Analysis_Data_Path + Read_Dict.iloc[6,2]
model3_save_path = Analysis_Data_Path + Read_Dict.iloc[7,2]
model4_save_path = Analysis_Data_Path + Read_Dict.iloc[8,2]
model5_save_path = Analysis_Data_Path + Read_Dict.iloc[9,2]
output_save_path = Analysis_Data_Path + Read_Dict.iloc[10,2]

models_save_path.append(model1_save_path)
models_save_path.append(model2_save_path)
models_save_path.append(model3_save_path)
models_save_path.append(model4_save_path)
models_save_path.append(model5_save_path)

New_data_dir = pd.read_csv(Analysis_Data_Path + 'New_data_list.csv',header=None)

# Determine use machine classified hate words or human classified hate words
while(True):
    method = input('Enter the processing method for dictionary: auto / manual: ')
    if str(method)=='auto':
        indicator = 0
        break
    elif str(method)=='manual':
        indicator = 1
        break

# Determine how many iterations
while(True):
    number_of_iteration = input('Number of Iterations: ')
    try:
        n = int(number_of_iteration)
        break
    except:
        number_of_iteration = input('Number of Iterations: ')

# Determine whether use the bigram method
while(True):
    if_two_gram = input('Use two grams method: Y/N? ')
    if if_two_gram=='Y':
        two_gram_indicator = True
        break
    else:
        two_gram_indicator = False
        break

Test_Data_Path = Clean_Data_Path + "TwitterTest_Benchmark_3.csv" # for LF

for i in range(n):
    print('Round: ',i+1)
    with open(output_save_path,'a') as f:
        f.write('Round: '+str(i+1))
        f.write('\n')
    if i != 0:
        old_data_path = old_data_save_path
        new_data_path = Raw_Data_Path + New_data_dir.iloc[i,0]
        if two_gram_indicator==False:
            old_dict_path = new_dict_save_path
        else:
            old_dict_path = new_two_gram_save_path
        old_data_save_path = old_data_save_path.replace(str(i)+'.csv',str(i+1)+'.csv')
        new_data_save_path = new_data_save_path.replace(str(i)+'.csv',str(i+1)+'.csv')
        prediction_save_path = prediction_save_path.replace(str(i)+'.csv',str(i+1)+'.csv')
        new_dict_save_path = new_dict_save_path.replace(str(i)+'.csv',str(i+1)+'.csv')
        new_two_gram_save_path = new_two_gram_save_path.replace(str(i)+'.csv',str(i+1)+'.csv')
        models_save_path[0] = models_save_path[0].replace(str(i)+'.sav',str(i+1)+'.sav')
        models_save_path[1] = models_save_path[1].replace(str(i)+'.sav',str(i+1)+'.sav')
        models_save_path[2] = models_save_path[2].replace(str(i),str(i+1))
        models_save_path[3] = models_save_path[3].replace(str(i)+'.sav',str(i+1)+'.sav')
        models_save_path[4] = models_save_path[4].replace(str(i)+'.sav',str(i+1)+'.sav')
        indicator=0

    old_dataset_relabelled = Emotion_Processing_Old(old_data_path,old_dict_path,indicator)  # change directory
    print('Old Dataset Relabelling Finished')
    Test_Data_Processed = Emotion_Processing_New(Test_Data_Path,old_dict_path,indicator)
    new_dataset_labelled = Emotion_Processing_New(new_data_path,old_dict_path,indicator)  # change directory
    print('New Dataset Labelling Finished')
    classification_result = Discriminator_Training(old_dataset_relabelled,new_dataset_labelled,Test_Data_Processed,models_save_path)
    print('Classifier Training Finished')
    Hate_dict = Dictionary_Update(classification_result,old_dict_path,indicator)  # change directory
    print('Dictionary Update Finished')
    classification_result = pd.read_csv(prediction_save_path)
    Hate_dict = pd.read_csv(new_dict_save_path)
    Two_Gram_Result = Two_Gram(classification_result,Hate_dict)
    print('Two Gram Dictionary Created')
