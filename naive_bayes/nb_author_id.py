#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""

import sys
sys.path.append('../')
from time import time
from sklearn.naive_bayes import GaussianNB
sys.path.append("D:\\GitHub\\ud120-projects\\tools")
from ml_local_tools.email_preprocess import preprocess


# features_train and features_test are the features for the training
# and testing datasets, respectively
# labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#########################################################
### your code goes here ###
t0 = time()
gnb = GaussianNB()
clf = gnb.fit(features_train, labels_train)
print "tempo de treinamento:", round(time()-t0, 3), "s"
t0 = time()
print(clf.score(features_test, labels_test))
print "tempo de treinamento:", round(time()-t0, 3), "s"

#########################################################
