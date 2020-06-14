import pandas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report

from nltk.corpus import stopwords
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer


def text_process(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


yelp = pd.read_csv('../data/yelp_academic_dataset_review.csv')
yelp = yelp[yelp['text'].notnull()]
yelp = yelp[:10000]
yelp['text length'] = yelp['text'].apply(len)
yelp_class = yelp
X = yelp_class['text']
y = yelp_class['stars']

bow_transformer = CountVectorizer(analyzer=text_process).fit(X)
X = bow_transformer.transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
nb = MultinomialNB()
nb.fit(X_train, y_train)
preds = nb.predict(X_test)
print(confusion_matrix(y_test, preds))
print('\n')
print(classification_report(y_test, preds))


# test_dataframe = pandas.read_csv('student_dn_annotations.txt', names=['word'])
test_dataframe = pandas.read_csv('outputpath/student_dn_annotations_freq_bigram.csv', names=['word', 'freq'])

for i, item in test_dataframe.iterrows():
    dish = item['word']
    dish_transformed = bow_transformer.transform([dish])
    pred = nb.predict(dish_transformed)[0]
    print('{},{},{}'.format(item['word'], item['freq'], str(pred)))

print('abc')