# %%

#!/usr/bin/python
def teste():
    import pickle
    import numpy
    ages = pickle.load( open("practice_outliers_ages.pkl", "r") )
    net_worths = pickle.load( open("practice_outliers_net_worths.pkl", "r") )

    ages       = numpy.reshape( numpy.array(ages), (len(ages), 1))
    net_worths = numpy.reshape( numpy.array(net_worths), (len(net_worths), 1))
    from sklearn.model_selection import train_test_split
    ages_train, ages_test, net_worths_train, net_worths_test = train_test_split(ages, net_worths, test_size=0.1, random_state=42)

    from sklearn.linear_model import LinearRegression
    reg = LinearRegression()
    reg.fit(ages_train, net_worths_train)
    predictions = reg.predict(ages_train)
    cleaned_data = outlierCleaner( predictions, ages_train, net_worths_train )
    print(cleaned_data[:10])

def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    import pandas as pd
    import math

    cleaned_data = []
    perc_clean = 0.1
    len_cleaned = int(math.floor((1 - perc_clean) * len(ages)))

    df_erros = pd.DataFrame({'ages': list(ages.flat), 'net_worths': list(net_worths.flat), 'predictions': list(predictions.flat)})

    df_erros['erros'] = ((df_erros['net_worths'] - df_erros['predictions'])).pow(2)
    df_erros.sort_values(by='erros', inplace=True)
    df_erros = df_erros[:len_cleaned]

    cleaned_data = zip(df_erros['ages'], df_erros['net_worths'], df_erros['erros'])

    return cleaned_data

#teste()
# %%
