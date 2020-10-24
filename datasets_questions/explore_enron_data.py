# %%
#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""
import os
import pickle

print(os.getcwd())
enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))



# %%
import pandas as pd

df = pd.DataFrame(enron_data).T
# %%
nan_pay = (df.total_payments == 'NaN').sum()
total = df.shape[0]
print('Total de pessoas: {}'.format(total))
print('Pessoas com NaN em pagamentos: {}'.format(nan_pay))
print('% com NaN em pagamentos: {}'.format(float(nan_pay)/float(total)))
# %%
df.columns
# %%
pois = df.poi.sum()
pois_nan_pay = (df[df.poi].total_payments.isnull()).sum()

print('Número de POIs: {}'.format(pois))
print('Número POIs com NaN em total_payment: {}'.format(pois_nan_pay))

# %%
df[df.poi].total_payments
# %%
