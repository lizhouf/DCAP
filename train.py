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
def neural_net(x):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)  # activation
    tf.nn.dropout(layer_1, 0.8)
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    out_layer = tf.nn.softmax(out_layer)
    return (out_layer)

def precision_voter(meta_classifier, conf_mtx):
    '''
    The meta-classifier decides the weight of each classifier based on precision 
    '''
    weight = np.empty((5,))
    for i in range(5):
        weight[i] = conf_mtx[i][0, 0] / (conf_mtx[i][0, 0] + conf_mtx[i][0, 1])
    weight = weight / np.sum(weight)
    output = meta_classifier * weight
    output = np.sum(output, axis=1)
    for i in range(output.shape[0]):
        if output[i] > 0.5:
            output[i] = 1
        else:
            output[i] = 0
    return output, weight

# Triain and Evaluate
def Discriminator_Training(df_old,df_new,Test_Data,models_save_path):
    '''
    This function is used to train the discriminators combining the old and new tweets.
    The function mainly includes 5 sub-models (logistic regression, support machine
    classifier, neural network, random forest classifier, and Gaussian Naive Bayes).
    After training the sub-models, there is a meta-model (bagging classifier) that vote
    for whether the tweet is a controversial speech based on the output of sub-models.
    Inputs:
      df_old: the data frame that contains the old tweets in the network;
      df_new: the data frame that contains the new tweets in the network;
      Test_Data: the data frame that contains the testing data; 
      models_save_path: the path to save models;
    Outputs:
      A txt file that contains all outputs in the output_save_path
    '''
    variables = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust', 'favorite_count',
                 'retweet_count','user_favourites_count', 'user_followers_count', 'user_friends_count', 'user_listed_count',
                 'number_hate_word']
    data = df_old
    data = data[data['maj_label'] != 'spam']  # Remove spams
    data_copy = data
    respond = list(data['maj_label'])
    data = data.loc[:, variables]
    data = np.asarray(data)
    variables = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust', 'favorite_count',
                 'retweet_count', 'user_favourites_count', 'user_followers_count', 'user_friends_count',
                 'user_listed_count', 'number_hate_word']

    for i in range(15):
        for j in range(data.shape[0]):
            if type(data[j, i]) == type('123'):
                try:
                    data[j, i] = float(data[j, i])
                except:
                    data[j, i] = 0
            if pd.isnull(data[j, i]) == True:
                data[j, i] = 0
    # Except for the sentiments, standardize all other columns
    for i in range(8, 15):
        std = StandardScaler()
        data[:, i] = std.fit_transform(data[:, i].reshape((-1, 1))).flatten()
    conf_mtx = []
    # Logistic Regression Model
    print("Model Training Started")

    respond_np = np.empty((len(respond),))
    for i in range(len(respond)):
        if type(respond[i]) == type('123'):
            if respond[i] == 'normal':
                respond_np[i] = 1
            else:
                respond_np[i] = 0
        elif type(respond[i])==type(123.0) or type(respond[i])==type(123):
            if respond[i]==1.0 or respond[i]==0.0:
                respond_np[i] = int(respond[i])
        else:
            respond_np[i] = 0
        respond_np[i] = int(respond_np[i])

    # Change all uncleaned data to 0 (avoid string in the dataset)
    x_train, x_test, y_train, y_test = train_test_split(data, respond_np, train_size=0.8)
    data_test = Test_Data.loc[:, variables]
    data_test = np.array(data_test)

    for i in range(15):
        for j in range(data_test.shape[0]):
            if type(data_test[j, i]) == type('123'):
                try:
                    data_test[j, i] = float(data_test[j, i])
                except:
                    data_test[j, i] = 0
            if pd.isnull(data_test[j, i]) == True:
                data_test[j, i] = 0
    data_test = np.array(data_test)
    # Except for the sentiments, standardize all other columns
    for i in range(8, 15):
        std = StandardScaler()
        data_test[:, i] = std.fit_transform(data_test[:, i].reshape((-1, 1))).flatten()

        # Save the output to csv file
    d = pd.concat([df_old.drop(['maj_label'],axis=1), df_new], ignore_index=True)
    ID = d.loc[:, 'id']
    tweet = d.loc[:, 'text']

    d_copy = d
    d = d.loc[:,variables]
    d = np.asarray(d)
    for i in range(15):
        for j in range(d.shape[0]):
            if type(d[j, i]) == type('123'):
                try:
                    d[j, i] = float(d[j, i])
                except:
                    d[j, i] = 0
            if pd.isnull(d[j, i]) == True:
                d[j, i] = 0
    for i in range(8, 15):
        std = StandardScaler()
        d[:, i] = std.fit_transform(d[:, i].reshape((-1, 1))).flatten()

    clf1 = LogisticRegression(C=1.0)
    clf1.fit(x_train, y_train)
    pickle.dump(clf1, open(models_save_path[0], 'wb'))
    y_pred = clf1.predict(x_test)
    print("Model 1 R2: ", clf1.score(x_test, y_test))
    print("Model 1 Confusion Matrix")
    print(confusion_matrix(y_test, y_pred))
    conf_mtx.append(confusion_matrix(y_test, y_pred))
    print()
    with open(output_save_path,'a') as f:
        f.write('Model 1 R2: '+str(clf1.score(x_test, y_test)))
        f.write('\n')
        f.write('Model 1 Confusion Matrix')
        f.write('\n')
        f.write(str(confusion_matrix(y_test,y_pred)))
        f.write('\n')
        f.write('\n')
    # Multi-SVM Model
    n_svc = 5  # Number of SVMs in the first layer
    f_svc = 8  # Features selected for each SVM
    clf2s = []
    samp_list = []
    for q in range(n_svc):
        samp = sample(range(15), f_svc)
        samp_list.append(samp)
        clf2 = SVC()
        clf2.fit(x_train[:, samp], y_train)
        pickle.dump(clf2, open(models_save_path[1].replace('.sav','sub_'+str(q)+'.sav'), 'wb'))
        clf2s.append(clf2)
    x_train_new = np.empty((x_train.shape[0], n_svc))
    for i in range(n_svc):
        pred = clf2s[i].predict(x_train[:, samp_list[i]])
        x_train_new[:, i] = pred.reshape((-1, 1)).flatten()
    meta_clf2 = SVC(C=1.0, kernel='rbf')
    meta_clf2.fit(x_train_new, y_train)
    pickle.dump(meta_clf2, open(models_save_path[1], 'wb'))

    x_test_new = np.empty((x_test.shape[0], n_svc))
    for i in range(n_svc):
        pred = clf2s[i].predict(x_test[:, samp_list[i]])
        x_test_new[:, i] = pred.reshape((-1, 1)).flatten()
    y_pred = meta_clf2.predict(x_test_new)
    print("Model 2 R2: ", meta_clf2.score(x_test_new, y_test))
    print("Model 2 Confusion Matrix")
    print(confusion_matrix(y_test, y_pred))
    conf_mtx.append(confusion_matrix(y_test, y_pred))
    with open(output_save_path,'a') as f:
        f.write('Model 2 R2: '+str(meta_clf2.score(x_test_new, y_test)))
        f.write('\n')
        f.write('Model 2 Confusion Matrix')
        f.write('\n')
        f.write(str(confusion_matrix(y_test,y_pred)))
        f.write('\n')
        f.write('\n')

    # ANN Model
    tf.set_random_seed(64)
    # Network Design
    num_input = 15
    X = tf.placeholder("float", [None, num_input])
    Y = tf.placeholder("float", [None, 2])
    weights = {
        'h1': tf.Variable(tf.random_normal([num_input, 32])),
        'out': tf.Variable(tf.random_normal([32, 2]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([32])),
        'out': tf.Variable(tf.random_normal([2]))
    }
    Y_hat = neural_net(X)
    loss_op = tf.losses.softmax_cross_entropy(Y, Y_hat)  # loss function
    optimizer = tf.train.AdamOptimizer(learning_rate=0.2)  # define optimizer # play around with learning rate
    train_op = optimizer.minimize(loss_op)  # minimize losss
    init = tf.global_variables_initializer()
    epoch = 40  # More Epoches Needed for Different Dataset
    y_train_new = np.empty((y_train.shape[0], 2))
    y_test_new = np.empty((y_test.shape[0], 2))
    for i in range(y_train.shape[0]):
        if y_train[i] == 0:
            y_train_new[i, 0] = 1
            y_train_new[i, 1] = 0
        else:
            y_train_new[i, 0] = 0
            y_train_new[i, 1] = 1
    for i in range(y_test.shape[0]):
        if y_test[i] == 0:
            y_test_new[i, 0] = 1
            y_test_new[i, 1] = 0
        else:
            y_test_new[i, 0] = 0
            y_test_new[i, 1] = 1
    with tf.Session() as sess:
        sess.run(init)
        for i in range(0, epoch):
            sess.run(train_op, feed_dict={X: x_train, Y: y_train_new})
            loss = sess.run(loss_op, feed_dict={X: x_train, Y: y_train_new})
            if (i % 20 == 0):
                print("Epoch No." + str(i) + " Entropy Loss: ", (loss))
        pred = sess.run(Y_hat, feed_dict={X: x_test})
        new_pred = np.empty((y_test.shape[0]))
        for i in range(pred.shape[0]):
            if pred[i, 0] == 0:
                new_pred[i] = 1
            else:
                new_pred[i] = 0
        test_loss = sess.run(loss_op, feed_dict={X: x_test, Y: y_test_new})
        pred_full = sess.run(Y_hat, feed_dict={X: data})
        pred_label = sess.run(Y_hat,feed_dict={X: d})
        pred_test = sess.run(Y_hat, feed_dict={X: data_test})
        saver = tf.train.Saver()
        saver.save(sess,models_save_path[2])
        print("Model 3 R2: ",accuracy_score(new_pred, y_test))
        print("Model 3 Confusion Matrix")
        print(confusion_matrix(y_test, new_pred))
        conf_mtx.append(confusion_matrix(y_test, new_pred))
        with open(output_save_path,'a') as f:
            f.write('Model 3 R2: '+str(accuracy_score(new_pred, y_test)))
            f.write('\n')
            f.write('Model 3 Confusion Matrix')
            f.write('\n')
            f.write(str(confusion_matrix(y_test,new_pred)))
            f.write('\n')
            f.write('\n')
    new_pred_full = np.empty((pred_full.shape[0]))
    for i in range(pred_full.shape[0]):
        if pred_full[i, 0] == 0:
            new_pred_full[i] = 0
        else:
            new_pred_full[i] = 1
    new_pred_label = np.empty((pred_label.shape[0]))
    for i in range(pred_label.shape[0]):
        if pred_label[i, 0] == 0:
            new_pred_label[i] = 0
        else:
            new_pred_label[i] = 1

    # Random Forest Classifier
    clf4 = RandomForestClassifier()
    clf4.fit(x_train, y_train)
    pickle.dump(clf4, open(models_save_path[3], 'wb'))
    pred = clf4.predict(x_test)
    print("Model 4 R2: ",clf4.score(x_test, y_test))
    print("Model 4 Confusion Matrix")
    print(confusion_matrix(y_test, pred))
    conf_mtx.append(confusion_matrix(y_test, pred))
    with open(output_save_path,'a') as f:
        f.write('Model 4 R2: '+str(clf4.score(x_test, y_test)))
        f.write('\n')
        f.write('Model 4 Confusion Matrix')
        f.write('\n')
        f.write(str(confusion_matrix(y_test,y_pred)))
        f.write('\n')
        f.write('\n')

    # Naive Bayes Classifier
    clf5 = GaussianNB()
    clf5.fit(x_train, y_train)
    pickle.dump(clf5, open(models_save_path[4], 'wb'))
    pred = clf5.predict(x_test)
    print("Model 5 R2: ", clf5.score(x_test, y_test))
    print("Model 5 Confusion Matrix")
    print(confusion_matrix(y_test, pred))
    conf_mtx.append(confusion_matrix(y_test, pred))
    with open(output_save_path,'a') as f:
        f.write('Model 5 R2: '+str(clf5.score(x_test, y_test)))
        f.write('\n')
        f.write('Model 5 Confusion Matrix')
        f.write('\n')
        f.write(str(confusion_matrix(y_test,y_pred)))
        f.write('\n')
        f.write('\n')


    # Create the input of the meta-classifier (output of base classifier)
    meta_classifier = np.empty((data.shape[0], 5))
    meta_classifier[:, 0] = clf1.predict(data).flatten()
    data_new = np.empty((data.shape[0], n_svc))
    for i in range(n_svc):
        pred = clf2s[i].predict(data[:, samp_list[i]])
        data_new[:, i] = pred.reshape((-1, 1)).flatten()
    meta_classifier[:, 1] = meta_clf2.predict(data_new).flatten()
    meta_classifier[:, 2] = new_pred_full
    meta_classifier[:, 3] = clf4.predict(data)
    meta_classifier[:, 4] = clf5.predict(data)

    output, weight = precision_voter(meta_classifier, conf_mtx)
    print('Final Result (Precision):')
    print("Meta Classifier R2: ",accuracy_score(respond_np, output))
    print("Meta Classifier Confusion Matrix")
    print(confusion_matrix(respond_np, output))
    with open(output_save_path,'a') as f:
        f.write('Final Result R2: '+str(accuracy_score(respond_np, output)))
        f.write('\n')
        f.write('Meta Classifier Confusion Matrix')
        f.write('\n')
        f.write(str(confusion_matrix(respond_np,output)))
        f.write('\n')
        f.write('\n')
    r_output = output 
    print()

    # Update Label
    meta_classifier = np.empty((d.shape[0], 5))
    meta_classifier[:, 0] = clf1.predict(d).flatten()
    data_new = np.empty((d.shape[0], n_svc))
    for i in range(n_svc):
        pred = clf2s[i].predict(d[:, samp_list[i]])
        data_new[:, i] = pred.reshape((-1, 1)).flatten()
    meta_classifier[:, 1] = meta_clf2.predict(data_new).flatten()
    meta_classifier[:, 2] = new_pred_label
    meta_classifier[:, 3] = clf4.predict(d)
    meta_classifier[:, 4] = clf5.predict(d)
    output, weight = precision_voter(meta_classifier, conf_mtx)
    r_output = output
    print(ID.shape)
    print(tweet.shape)
    print(r_output.shape)
    result = pd.DataFrame({'ID': np.array(ID), 'Tweet': tweet, 'Is_Hate': r_output})
    result.to_csv(prediction_save_path)
    old_data_change = d_copy
    d_copy['maj_label'] = r_output
    d_copy.to_csv(old_data_save_path)

    ## Test Part
    y_test = list(Test_Data['is_hate'])
    for i in y_test:
        if i == 1:
            i = 0
        else:
            i = 1
    new_pred_full = np.empty((pred_test.shape[0]))
    for i in range(pred_test.shape[0]):
        if pred_test[i, 0] == 0:
            new_pred_full[i] = 0
        else:
            new_pred_full[i] = 1
    meta_classifier = np.empty((data_test.shape[0], 5))
    meta_classifier[:, 0] = clf1.predict(data_test).flatten()
    data_new = np.empty((data_test.shape[0], n_svc))
    for i in range(n_svc):
        pred = clf2s[i].predict(data_test[:, samp_list[i]])
        data_new[:, i] = pred.reshape((-1, 1)).flatten()
    meta_classifier[:, 1] = meta_clf2.predict(data_new).flatten()
    meta_classifier[:, 2] = new_pred_full
    meta_classifier[:, 3] = clf4.predict(data_test)
    meta_classifier[:, 4] = clf5.predict(data_test)
    output, weight = precision_voter(meta_classifier, conf_mtx)
    r_output = output
    print('Test Score for this Round:', accuracy_score(r_output,y_test))
    print('Confusion Matrix for Test Data:')
    print(confusion_matrix(y_test,r_output))
    with open(output_save_path,'a') as f:
        f.write('Test Score for this Round: '+str(accuracy_score(r_output, y_test)))
        f.write('\n')
        f.write('Confusion Matrix for Test Data:')
        f.write('\n')
        f.write(str(confusion_matrix(y_test,r_output)))
        f.write('\n')
        f.write('\n')

    return result
