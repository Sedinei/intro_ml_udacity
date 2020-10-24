#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""

import sys
sys.path.append('../')
from sklearn.svm import SVC
from time import time
import matplotlib.pyplot as plt
from ml_local_tools.email_preprocess import preprocess

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#features_train = features_train[:len(features_train)/100]
#labels_train = labels_train[:len(labels_train)/100]

print('C=10000')
t0 = time()
clf = SVC(kernel='rbf', C=10000)
clf.fit(features_train, labels_train)
print "tempo de treinamento:", round(time()-t0, 3), "s"
#t0 = time()
#print(clf.score(features_test, labels_test))
#print "tempo de teste:", round(time()-t0, 3), "s"
#Predições
pred = clf.predict(features_test)
print('Previsão para o registro 10: {}'.format(pred[10]))
print('Previsão para o registro 26: {}'.format(pred[26]))
print('Previsão para o registro 50: {}'.format(pred[50]))
print('Total de previsões: {}'.format(len(pred)))
print('Total classificado como Chris (1): {}'.format(pred.sum()))
