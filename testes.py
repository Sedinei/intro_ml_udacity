# %%
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import pickle
# %%
word_data = pickle.load(open("your_word_data.pkl", 'r'))
# %%
word_data[:10]
# %%
stpwrds = stopwords.words('english')
# %%
stpwrds[:10]
# %%
stpwrds.index('ve')
# %%
vectorizer = TfidfVectorizer(stop_words='english')
# %%
word_vec = vectorizer.fit(word_data)
# %%
len(vectorizer.get_feature_names())
# %%
vectorizer.get_feature_names()[34597]
# %%
