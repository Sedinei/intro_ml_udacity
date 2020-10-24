# %%
import sys
sys.path.append('../')
from ml_local_tools.class_vis import prettyPicture
from ml_local_tools.prep_terrain_data import makeTerrainData

import matplotlib.pyplot as plt
import copy
import numpy as np
import pylab as pl


features_train, labels_train, features_test, labels_test = makeTerrainData()


########################## SVM #################################
### we handle the import statement and SVC creation for you here
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf.fit(features_train, labels_train)
print(clf.score(features_test, labels_test))

#### now your job is to fit the classifier
#### using the training features/labels, and to
#### make a set of predictions on the test data
prettyPicture(clf, features_test, labels_test, 'DT')


#### store your predictions in a list named pred
pred = clf.predict(features_test)




from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)

def submitAccuracy():
    return acc
# %%

# %%