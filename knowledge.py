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

# helpers
def Prob_Word_Hate(hate_count, word, num_hate):
    if word not in hate_count.keys():
        return 0
    elif hate_count[word] < 5:
        return 0
    else:
        return hate_count[word] / num_hate

def Prob_Word(full_count, word, num_tweet):
    if word not in full_count.keys():
        return 0
    elif full_count[word] < 5:
        return 0
    else:
        return full_count[word] / num_tweet

# Update Knowledge
def Dictionary_Update(Predicted_Result, Original_Hate,indicator=0):
    '''
    This function is used to update the dictionaries (serves as generator).
    The new dictionary is calculated based on the tf-idf scores, the frequency of
    hate words appeared in controversial speech.
    Inputs:
      Predicted_Result: the data frame contains the predicted results of the tweets;
      Original_Hate: the csv file that contains the hate words from the previous round;
      indicator: 0, use machine classified hate words;
                 1, use human classified hate words;
    Outputs:
      A csv dictionary containing the new hate words, confidence scores, and Is_new
      indicating whether the word is a new hate word in the directory new_dict_save_path.
    '''

    filepath = Raw_Data_Path+'28K adjectives.txt'
    with open(filepath) as fp:
        adjectives = fp.read()
    adjectives = adjectives + ' https' + ' http' + ' hashtag'

    filepath = Raw_Data_Path+'31K verbs.txt'
    with open(filepath) as fp:
        verbs = fp.read()
    filepath = Raw_Data_Path+'91K nouns.txt'
    with open(filepath) as fp:
        nouns = fp.read()

    hate_words = pd.read_csv(Original_Hate)
    if indicator==1:
        hate_words = hate_words[hate_words['Is_keep']==1]
    df_doc = Predicted_Result
    df_doc["tweet_clean"] = df_doc['Tweet'].apply(lambda x: x.lower() if type(x) == type('123') else x)
    df_doc["tweet_clean"] = df_doc['Tweet'].apply(
        lambda x: re.sub(r'[^a-zA-Z]|(\w+:\/\/\S+)', ' ', x) if type(x) == type('123') else x)
    words = []
    words_hate = []

    for i in range(df_doc.shape[0]):
        speech = df_doc['tweet_clean'].iloc[i]  # tweet
        if type(speech) != type('123'):
            continue
        elif df_doc['Is_Hate'].iloc[i] == 0:
            doc = nlp(speech)
            for token in doc:
                if token.pos_ in ["NOUN", "VERB", "ADJ"]:
                    words.append(token.text)
                    words_hate.append(token.text)
        else:
            doc = nlp(speech)
            for token in doc:
                if token.pos_ in ["NOUN", "VERB", "ADJ"]:
                    words.append(token.text)

    words = [x.lower() for x in words]
    words_hate = [x.lower() for x in words_hate]
    words_full = words
    words = set(words)
    words = list(words)
    words_new = []
    for i in range(len(words)):
        if words[i] not in adjectives and words[i] not in verbs and words[i] not in nouns:
            words_new.append(words[i])
    prob = np.empty((hate_words.shape[0],))
    df_doc = df_doc.drop_duplicates(['tweet_clean'])
    words_full_count = collections.Counter(words_full)
    words_hate_count = collections.Counter(words_hate)
    num_tweet = 0
    num_hate = 0
    for i in range(df_doc.shape[0]):
        speech = df_doc['tweet_clean'].iloc[i]  # tweet
        if type(speech) != type('123'):
            continue
        elif df_doc['Is_Hate'].iloc[i] == 0:
            num_tweet += 1
            num_hate += 1
        else:
            num_tweet += 1
    P_A = num_hate / num_tweet
    old_words = list(hate_words['Word'])
    words = [i for i in words if i not in old_words]

    for i in range(hate_words.shape[0]):
        word = hate_words['Word'].iloc[i]
        P_B_A = Prob_Word_Hate(words_hate_count, word, num_hate)
        P_B = Prob_Word(words_full_count, word, num_tweet)
        if P_B == 0:
            prob[i] = 0
        else:
            prob[i] = P_A * P_B_A / P_B
    prob_new = np.empty((len(words),))
    for i in range(len(words)):
        word = words[i]
        P_B_A = Prob_Word_Hate(words_hate_count, word, num_hate)
        P_B = Prob_Word(words_full_count, word, num_tweet)
        if P_B == 0:
            prob_new[i] = 0
        else:
            prob_new[i] = P_A * P_B_A / P_B

    hate_words_list = list(hate_words['Word'])
    hate_prob = list(prob)

    for i in range(prob_new.shape[0]):
        if prob_new[i] >= 4 * P_A:
            hate_words_list.append(words[i])
            hate_prob.append(prob_new[i])
    is_new = np.empty((len(hate_prob),))
    for i in range(is_new.shape[0]):
        if i < hate_words.shape[0]:
            is_new[i] = 0
        else:
            is_new[i] = 1
    print('Old Hate Words: ', len(hate_words))
    print('New Hate Words: ', len(hate_prob) - len(hate_words))
    print('Total Hate Words: ', len(hate_prob))
    with open(output_save_path,'a') as f:
        f.write('Old Hate Words: '+str(len(hate_words)))
        f.write('\n')
        f.write('New Hate Words: '+str(len(hate_prob)-len(hate_words)))
        f.write('\n')
        f.write('Total Hate Words: '+str(len(hate_prob)))
        f.write('\n')
        f.write('\n')
    result = pd.DataFrame({'Word': hate_words_list, 'Score': hate_prob, 'Is_new': is_new})
    result.to_csv(new_dict_save_path)
    return result

# Update Knowledge
def Two_Gram(Predicted_Result, Hatebase_Score):
    '''
    This function is used to update the dictionaries (serves as generator) instead of by
    unigrams but by bi-grams. That is, assuming all hate words as bigrams and determining
    which bigram belongs to hate dictionary based on hate scores of unigrams.
    Inputs:
      Predicted_Result: the data frame contains the tweets classified by discriminators;
      Hatebase_Score: the hate word dictionary generated from Dictionary_Update;
    Outputs:
      A csv file containing the bigram hate words dictionary includes the bigram and
      confidence score;
    '''
    df_doc = Predicted_Result
    hate_dict = Hatebase_Score
    df_doc["tweet_clean"] = df_doc['Tweet'].apply(lambda x: x.lower() if type(x) == type('123') else x)
    df_doc["tweet_clean"] = df_doc['Tweet'].apply(
        lambda x: re.sub(r'[^a-zA-Z]|(\w+:\/\/\S+)', ' ', x) if type(x) == type('123') else x)

    hate_words = list(hate_dict['Word'])
    hate_score = list(hate_dict['Score'])
    hate_dict = {}
    for i in range(len(hate_words)):
        hate_dict[hate_words[i]] = hate_score[i]
    two_gram_list = []
    two_gram_score = []
    for i in range(df_doc.shape[0]):
        speech = df_doc['tweet_clean'].iloc[i]  # tweet
        if type(speech) != type('123'):
            continue
        else:
            doc = nlp(speech)
            token_list = []
            for token in doc:
                token_list.append(token)
        for j in range(len(token_list) - 1):
            if str(token_list[j]) in hate_words and str(token_list[j + 1]) in hate_words:
                two_gram = str(token_list[j]) + ' ' + str(token_list[j + 1])
                if two_gram not in two_gram_list:
                    two_gram_list.append(str(token_list[j]) + ' ' + str(token_list[j + 1]))
                    score1 = hate_dict[str(token_list[j])]
                    score2 = hate_dict[str(token_list[j + 1])]
                    two_gram_score.append(score1 * score2)

    print("Number of Two Grams: ", len(two_gram_list))
    with open(output_save_path,'a') as f:
        f.write("Number of Two Grams: "+str(len(two_gram_list)))
        f.write('\n')
        f.write('\n')
    result = pd.DataFrame({'Gram': two_gram_list, 'Score': two_gram_score})
    result.to_csv(new_two_gram_save_path)
    return result
