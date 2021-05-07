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

# process in seed round
def Emotion_Processing_Old(filename,dict_name,indicator=0):
    '''
    This function is used to process data that is originally in the network.
    The function mainly includes counting the number of hate words according to the 
    newly updated hate word dictionary.
    Inputs:
      filename: the filename that includes the old tweets.
      dict_name: the filename of the hate dictionary after the previous round.
      indicator: 0, use the machine classified dictionary;
                 1, use the human classified dictionary;
    Outputs:
      Tweets that are originally in the network but with updated count of hate words
      saved in the position old_data_save_path
    '''
    df = pd.read_csv(filename)
    print("Start Preprocessing Emotion")
    try:
        df["tweet_clean"] = df.text.apply(lambda x:str(x).lower())
        df["tweet_clean"] = df.text.apply(lambda x:re.sub(r'[^a-zA-Z]|(\w+:\/\/\S+)',' ',str(x)))
        df['word_count'] = df['tweet_clean'].apply(lambda x: x.count(" ")+1)
    except:
        df["tweet_clean"] = df.tweet.apply(lambda x:str(x).lower())
        df["tweet_clean"] = df.tweet.apply(lambda x:re.sub(r'[^a-zA-Z]|(\w+:\/\/\S+)',' ',str(x)))
        df['word_count'] = df['tweet_clean'].apply(lambda x: x.count(" ")+1)
    print("Start Aspect-based Emotion Analysis and Hate Speech Analysis")
    filepath = (Raw_Data_Path+'NRC-Emotion-Lexicon-Wordlevel-v0.92.txt')
    emolex_df = pd.read_csv(filepath,
                        names=["word", "emotion", "association"],
                        sep='\t')
    hate_dic = pd.read_csv(dict_name)
    if indicator==1:
        hate_dic = hate_dic[hate_dic['Is_keep']==1]
    hate_dic["len"]= hate_dic.Word.apply(lambda x: x.count(" "))
    hate_dic = hate_dic[hate_dic.len==0].reset_index(drop=True)
    hate_words_list = list(hate_dic.Word)
    hate_words_score = list(hate_dic.Score)
    number_hate_word = [] 
    column = "tweet_clean"

    for i in range(len(df)):
        document = word_tokenize(df.loc[i][column])
        hate_count = 0

        for word in document:
            word = stemmer.stem(word.lower())
            if word in hate_words_list:
                ind = hate_words_list.index(word)
                if hate_words_score[ind]==0:
                    hate_count += 1
                else:
                    hate_count += hate_words_score[ind]
        number_hate_word.append(hate_count)

    df["number_hate_word"]=number_hate_word
    df.to_csv(old_data_save_path,index=False)
    print("Emotion Preprocessing Finished.")
    return df

# process in later rounds
def Emotion_Processing_New(filename,dict_name,indicator=0):
    '''
    This function is used to process data that is newly fed into the network.
    The function mainly includes the sentiment analysis.
    Inputs:
      filename: the filename that includes the new tweets.
      dict_name: the filename of the hate dictionary after the previous round.
      indicator: 0, use the machine classified dictionary;
                 1, use the human classified dictionary;
    Outputs:
      post-sentiment analysis tweets saved in the position new_data_save_path
    '''
    # note: this code is for mac os, window systems may use different encodings
    df = pd.read_csv(filename,nrows = 1000, encoding="ISO-8859-1")
    if df.shape[0]>=50000:
        samples = random.sample(range(df.shape[0]),100000)
        df = df.iloc[samples,:]
    print("Start Preprocessing Emotion")
    df["tweet_clean"] = df.text.apply(lambda x:str(x).lower())
    df["tweet_clean"] = df.text.apply(lambda x:re.sub(r'[^a-zA-Z]|(\w+:\/\/\S+)',' ',str(x)))
    df['word_count'] = df['tweet_clean'].apply(lambda x: x.count(" ")+1)
    print("Start Aspect-based Emotion Analysis and Hate Speech Analysis")
    filepath = (Raw_Data_Path+'NRC-Emotion-Lexicon-Wordlevel-v0.92.txt')
    emolex_df = pd.read_csv(filepath,
                        names=["word", "emotion", "association"],
                        sep='\t')
    anger_dic = emolex_df[emolex_df.emotion=="anger"].drop(columns=["emotion"]).set_index("word").T.to_dict('dict')
    anticipation_dic = emolex_df[emolex_df.emotion=="anticipation"].drop(columns=["emotion"]).set_index("word").T.to_dict('dict')
    disgust_dic = emolex_df[emolex_df.emotion=="disgust"].drop(columns=["emotion"]).set_index("word").T.to_dict('dict')
    fear_dic = emolex_df[emolex_df.emotion=="fear"].drop(columns=["emotion"]).set_index("word").T.to_dict('dict')
    joy_dic = emolex_df[emolex_df.emotion=="joy"].drop(columns=["emotion"]).set_index("word").T.to_dict('dict')
    sadness_dic = emolex_df[emolex_df.emotion=="sadness"].drop(columns=["emotion"]).set_index("word").T.to_dict('dict')
    surprise_dic = emolex_df[emolex_df.emotion=="surprise"].drop(columns=["emotion"]).set_index("word").T.to_dict('dict')
    trust_dic = emolex_df[emolex_df.emotion=="trust"].drop(columns=["emotion"]).set_index("word").T.to_dict('dict')
    positive_dic = emolex_df[emolex_df.emotion=="positive"].drop(columns=["emotion"]).set_index("word").T.to_dict('dict')
    negative_dic = emolex_df[emolex_df.emotion=="negative"].drop(columns=["emotion"]).set_index("word").T.to_dict('dict')
    anger_list = []
    anticipation_list = []
    disgust_list = []
    fear_list = []
    joy_list = []
    sadness_list = []
    surprise_list = []
    trust_list = []
    positive_list = []
    negative_list = []

    hate_dic = pd.read_csv(dict_name)
    if indicator==1:
        hate_dic = hate_dic[hate_dic['Is_keep']==1]
    hate_dic["len"]= hate_dic.Word.apply(lambda x: x.count(" "))
    hate_dic = hate_dic[hate_dic.len==0].reset_index(drop=True)
    hate_words_list = list(hate_dic.Word)
    hate_words_score = list(hate_dic.Score)
    number_hate_word = [] # 0 for not contain
    column = "tweet_clean"

    for i in range(len(df)):
        document = word_tokenize(df.loc[i][column])
        anger_score = 0
        anticipation_score = 0
        disgust_score = 0
        fear_score = 0
        joy_score = 0
        sadness_score = 0
        surprise_score = 0
        trust_score = 0
        positive_score = 0
        negative_score = 0
        hate_count = 0

        for word in document:
            word = stemmer.stem(word.lower())
            try:
                anger_score += anger_dic[word]['association']
                anticipation_score += anticipation_dic[word]['association']
                disgust_score += disgust_dic[word]['association']
                fear_score += fear_dic[word]['association']
                joy_score += joy_dic[word]['association']
                sadness_score += sadness_dic[word]['association']
                surprise_score += surprise_dic[word]['association']
                trust_score += trust_dic[word]['association']
                positive_score += positive_dic[word]['association']
                negative_score += negative_dic[word]['association']
            except:
                0
            if word in hate_words_list:
                ind = hate_words_list.index(word)
                if hate_words_score[ind] == 0:
                    hate_count += 1
                else:
                    hate_count += hate_words_score[ind]
        anger_list.append(anger_score)
        anticipation_list.append(anticipation_score)
        disgust_list.append(disgust_score)
        fear_list.append(fear_score)
        joy_list.append(joy_score)
        sadness_list.append(sadness_score)
        surprise_list.append(surprise_score)
        trust_list.append(trust_score)
        positive_list.append(positive_score)
        negative_list.append(negative_score)
        number_hate_word.append(hate_count)

    df["anger"] = anger_list
    df["anticipation"] = anticipation_list
    df["disgust"] = disgust_list
    df["fear"] = fear_list
    df["joy"] = joy_list
    df["sadness"] = sadness_list
    df["surprise"] = surprise_list
    df["trust"] = trust_list
    df["positive"] = positive_list
    df["negative"] = negative_list
    df["number_hate_word"]=number_hate_word
    df.to_csv(new_data_save_path,index=False)
    print("Emotion Preprocessing Finished.")
    return df
