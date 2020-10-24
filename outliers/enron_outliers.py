# %%
% matplotlib inline
#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../")
from ml_local_tools.feature_format import featureFormat, targetFeatureSplit

def find_outlier(data_dict, feature, min_value):
    outliers = []
    for name in data_dict:
        if data_dict[name][feature] == 'NaN': continue
        elif data_dict[name][feature] > min_value:
            outliers.append((name, data_dict[name][feature]))

    return outliers

### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
data_dict.pop('TOTAL', 0)
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)

### your code below
#print(find_outlier(data_dict, 'salary', 25000000))

for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()



# %%
