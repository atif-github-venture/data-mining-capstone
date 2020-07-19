# inspiration - https://github.com/jonchang03/cs598-dm-capstone/
import sys

from Task6.UtilWordEmbedding import MeanEmbeddingVectorizer

sys.maxsize > 2**32
import pandas as pd
import numpy as np
import multiprocessing
import gensim
import nltk
import spacy
# from UtilWordEmbedding import MeanEmbeddingVectorizer

from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

from gensim.parsing.preprocessing import strip_tags, strip_punctuation, strip_numeric, strip_short
from gensim.parsing.preprocessing import strip_multiple_whitespaces, strip_non_alphanum, remove_stopwords, stem_text
from gensim.models.word2vec import Word2Vec
from nltk.stem import WordNetLemmatizer, SnowballStemmer

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn import svm
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, GridSearchCV
from tqdm import tqdm

hygiene_text_path = "./Hygiene/hygiene.dat"
hygiene_labels_path = "./Hygiene/hygiene.dat.labels"
hygiene_additional_path = "./Hygiene/hygiene.dat.additional"

from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words('english'))

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
SEED = 26

FILTERS_LIST = [lambda x: x.lower(),  # lowercase
                strip_tags,  # remove tags
                strip_punctuation,  # replace punctuation characters with spaces
                strip_multiple_whitespaces,  # remove repeating whitespaces
                # strip_numeric, # remove numbers
                gensim.parsing.preprocessing.remove_stopwords,  # remove stopwords
                strip_short,  # remove words less than minsize=3 characters long]
                stem_text]


def preprocess(text):
    """
    strip_tags, strip_punctuation, strip_multiple_whitespaces, strip_numeric,
    """
    result_stemmed = []
    for token in gensim.parsing.preprocessing.preprocess_string(text, FILTERS_LIST):
        result_stemmed.append(WordNetLemmatizer().lemmatize(token))
    return result_stemmed


texts = []
preprocessed_texts = []

with open(hygiene_text_path) as f:
    texts = f.readlines()

for _text in tqdm(texts):
    result_stemmed = preprocess(_text)
    preprocessed_texts.append(result_stemmed)

all_preprocessed_texts = [" ".join(_text) for _text in preprocessed_texts]

N = 546

# labels
with open(hygiene_labels_path, 'r') as f:
    labels = [l.rstrip() for l in f]

# texts = []
# with open(hygiene_text_path, 'r') as f:
#     texts = f.read().splitlines(True)


df = pd.DataFrame({"label":labels, "text": texts,
                   "preprocessed_texts": all_preprocessed_texts,
                   "tokenized_texts": preprocessed_texts})
hygiene_additional = pd.read_csv(hygiene_additional_path,
                                 names=["cuisines_offered", "zipcode", "num_reviews", "avg_rating"],
                                 dtype={"cuisines_offered": str,
                                        "zipcode": str,
                                        "num_reviews": str})
df = df.join(hygiene_additional)
df['avg_rating'] = df['avg_rating'].apply(lambda x: str(int(round(x, 0))))

print(df.info())
print(df.head())

train_df = df[df["label"] != "[None]"]
test_df = df[df["label"] == "[None]"]

additional_feats = ["cuisines_offered", "zipcode", "num_reviews", "avg_rating"]

train = train_df[["text"] + additional_feats]
train_preprocessed = train_df[["preprocessed_texts"] + additional_feats]
train_tokenized = train_df[["tokenized_texts"] + additional_feats]
train_labels = train_df["label"].astype(int) # needed by sklearn

test = test_df[["text"] + additional_feats]
test_preprocessed = test_df[["preprocessed_texts"] + additional_feats]
test_tokenized = test_df[["tokenized_texts"] + additional_feats]
test_labels = test_df["label"]

print(train.shape, train_preprocessed.shape, train_tokenized.shape, train_labels.shape)
print(test.shape, test_preprocessed.shape, test_tokenized.shape, test_labels.shape)
print(train.dtypes, train_preprocessed.dtypes, train_tokenized.dtypes)

print(train.head())
print(train_preprocessed.head())
print(train_tokenized.head())

from sklearn import preprocessing

pipeline = Pipeline([
    ('preprocess', ColumnTransformer(
        [('cuisines_offered', CountVectorizer(min_df=10), 'cuisines_offered'),
         ('zipcode', OneHotEncoder(dtype='int', handle_unknown='ignore'), ['zipcode']),
         ('num_reviews', CountVectorizer(max_df=7, token_pattern='\d+'), 'num_reviews'),
         ('avg_rating', OneHotEncoder(dtype='int', handle_unknown='ignore'), ['avg_rating']),
         ('text', TfidfVectorizer(
             stop_words='english',
             strip_accents='unicode',
             min_df=3,
             max_df=0.5,
             ngram_range=(1, 3),
             max_features=500), 'preprocessed_texts')],
        remainder='passthrough',
    )),
    ('clf', MultinomialNB())
], verbose=False)

# pipeline.fit(X_train, y_train)
# y_pred = pipeline.predict(X_test)
# scores = metrics.f1_score(y_test, y_pred)
scores = cross_val_score(pipeline, train_preprocessed, train_labels, cv=5, scoring= 'f1_macro')
print(scores)
print("Average F1-Score: %0.5f" % np.average(scores))

from sklearn import preprocessing

pipeline = Pipeline([
    ('preprocess', ColumnTransformer(
        [('cuisines_offered', CountVectorizer(min_df=10), 'cuisines_offered'),
         ('zipcode', OneHotEncoder(dtype='int', handle_unknown='ignore'), ['zipcode']),
         ('num_reviews', CountVectorizer(max_df=7, token_pattern='\d+'), 'num_reviews'),
         ('avg_rating', OneHotEncoder(dtype='int', handle_unknown='ignore'), ['avg_rating']),
         ('text', TfidfVectorizer(
             stop_words='english',
             strip_accents='unicode',
             min_df=3,
             max_df=0.5,
             ngram_range=(1, 3),
             max_features=500), 'text')],
        remainder='passthrough',
    )),
    ('clf', MultinomialNB())
], verbose=False)

# pipeline.fit(X_train, y_train)
# y_pred = pipeline.predict(X_test)
# scores = metrics.f1_score(y_test, y_pred)
scores = cross_val_score(pipeline, train, train_labels, cv=5, scoring= 'f1')
print(scores)
print("Average F1-Score: %0.5f" % np.average(scores))

print(len(df['num_reviews'].value_counts()))
df['num_reviews'].value_counts()

print(len(df['cuisines_offered'].value_counts()))
df['cuisines_offered'].value_counts()

print(len(df['avg_rating'].value_counts()))
print(len(df['zipcode'].value_counts()))

pipeline = Pipeline([
    ('preprocess', ColumnTransformer(
        [('cuisines_offered', CountVectorizer(), 'cuisines_offered'),
         ('zipcode', OneHotEncoder(dtype='int', handle_unknown='ignore'), ['zipcode']),
         ('num_reviews', CountVectorizer(token_pattern='\d+'), 'num_reviews'),
         ('avg_rating', CountVectorizer(token_pattern='\d+'), 'avg_rating'),
         ('text', TfidfVectorizer(
             stop_words='english',
             strip_accents='unicode',
             min_df=3,
             max_df=0.5,
             ngram_range=(1, 3),
             max_features=500), 'preprocessed_texts')],
        remainder='passthrough',
    )),
    ('clf', MultinomialNB())
], verbose=False)

# pipeline.fit(X_train, y_train)
# y_pred = pipeline.predict(X_test)
# scores = metrics.f1_score(y_test, y_pred)
scores = cross_val_score(pipeline, train_preprocessed, train_labels, cv=5, scoring= 'f1')
print(scores)
print("Average F1-Score: %0.5f" % np.average(scores))

pipeline = Pipeline([
    ('union', ColumnTransformer(
        [('cuisines_offered', CountVectorizer(), 'cuisines_offered'),
         ('zipcode', OneHotEncoder(dtype='int', handle_unknown='ignore'), ['zipcode']),
         ('num_reviews', CountVectorizer(token_pattern='\d+'), 'num_reviews'),
         ('avg_rating', CountVectorizer(token_pattern='\d+'), 'avg_rating'),
         ('text', TfidfVectorizer(
             stop_words='english',
             strip_accents='unicode',
             min_df=15,
             max_df=0.5,
             ngram_range=(1, 3),
             max_features=500), 'text')],
        remainder='passthrough',
    )),
    ('clf', MultinomialNB())
], verbose=False)

# pipeline.fit(X_train, y_train)
# y_pred = pipeline.predict(X_test)
# score = metrics.f1_score(y_test, y_pred)
scores = cross_val_score(pipeline, train, train_labels, cv=5, scoring= 'f1')
print(scores)
print("Average F1-Score: %0.5f" % np.average(scores))


# Create Function for Testing
def test_classifier(clf, X, y, vectorizer, text_col='text'):
    pipeline = Pipeline([
        ('union', ColumnTransformer(
            [('cuisines_offered', CountVectorizer(min_df=10), 'cuisines_offered'),
             ('zipcode', OneHotEncoder(dtype='int', handle_unknown='ignore'), ['zipcode']),
             ('num_reviews', CountVectorizer(max_df=7, token_pattern='\d+'), 'num_reviews'),
             ('avg_rating', OneHotEncoder(dtype='int', handle_unknown='ignore'), ['avg_rating']),
             ('text', vectorizer, text_col)],
            remainder='passthrough',
        )),
        ('clf', clf)
    ], verbose=False)
    scores = cross_val_score(pipeline, X, y, cv=5, scoring= 'f1_macro')
    print(clf)
    print(scores)
    cv_score = np.average(scores)
    return cv_score

classifiers = {
    'Naive Bayes': MultinomialNB(),
    'Support Vector Machine': svm.SVC(),
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(random_state=SEED, n_estimators=500, n_jobs=-1),
    #'Gradient Boosting': GradientBoostingClassifier()
    'XGBoost': XGBClassifier(n_estimators=500,
                             max_depth=5,
                             learning_rate=0.2,
                             objective='binary:logistic',
                             scale_pos_weight=2,
                             n_jobs=-1,
                             random_state=SEED)
}

tfidf = TfidfVectorizer(
    stop_words='english',
    strip_accents='unicode',
    min_df=3,
    max_df=0.5,
    ngram_range=(1, 3),
    max_features=500)
bow = CountVectorizer(
    stop_words=STOP_WORDS,
    strip_accents='unicode',
    min_df=15,
    max_df=0.5,
    ngram_range=(1, 3))


# BOW - No Preprocessing
for clf_name, clf in classifiers.items():
    cv_score = test_classifier(clf, train, train_labels,
                               vectorizer=bow, text_col='text')
    print('{}: {}'.format(clf_name, cv_score))

# BOW - Preprocessing
for clf_name, clf in classifiers.items():
    cv_score = test_classifier(clf, train_preprocessed, train_labels,
                               vectorizer=bow, text_col='preprocessed_texts')
    print('{}: {}'.format(clf_name, cv_score))

# TFIDF - No Preprocessing
for clf_name, clf in classifiers.items():
    cv_score = test_classifier(clf, train, train_labels,
                               vectorizer=tfidf, text_col='text')
    print('{}: {}'.format(clf_name, cv_score))

# TFIDF - Preprocessing

for clf_name, clf in classifiers.items():
    cv_score = test_classifier(clf, train_preprocessed, train_labels,
                               vectorizer=tfidf, text_col='preprocessed_texts')
    print('{}: {}'.format(clf_name, cv_score))


# word2vec embeddings
num_workers = multiprocessing.cpu_count()
print(num_workers)

word_model = Word2Vec(preprocessed_texts,
                      size=300, window=5, min_count=3,
                      workers=num_workers)
w2v = {w: vec for w, vec in zip(word_model.wv.index2word, word_model.wv.syn0)}
mean_vec_tr = MeanEmbeddingVectorizer(word_model)
w2v_embeddings = mean_vec_tr.transform(preprocessed_texts)

w2v_train = w2v_embeddings[:546]
w2v_test = w2v_embeddings[546:]
print(w2v_train.shape)
print(w2v_test.shape)

print(train_df.shape)
print(test_df.shape)

w2v_train = w2v_embeddings[:546]
clf = svm.SVC()# XGBClassifier(n_estimators=500, n_jobs=-1)
scores = cross_val_score(clf, w2v_train, train_labels, cv=5, scoring= 'f1')
print(scores)
print("Average F1-Score: %0.5f" % np.average(scores))

w2v_train = w2v_embeddings[:546]
params = {'max_depth': 6,
          'eta': 0.3,
          'objective': 'binary:logistic',
          'subsample':0.8,
          'n_estimators': 500,
          'scale_pos_weight': 2,
          'eval_metric': 'auc',
          'n_jobs': -1}
clf = XGBClassifier(n_estimators=500,
                    max_depth=5,
                    learning_rate=0.2,
                    objective='binary:logistic',
                    scale_pos_weight=2,
                    n_jobs=-1,
                    random_state=SEED)
scores = cross_val_score(clf, w2v_train, train_labels, cv=5, scoring= 'f1')
print(scores)
print("Average F1-Score: %0.5f" % np.average(scores))

# Ensemble Using mlens
# https://mlens.readthedocs.io/en/0.1.x/ensemble_tutorial/#ensemble-model-selection

from mlens.ensemble import SuperLearner
from sklearn.metrics import f1_score

def f1(y, p): return f1_score(y, p, average='macro')

# --- Build ---

# Passing a scoring function will create cv scores during fitting
# the scorer should be a simple function accepting to vectors and returning a scalar
ensemble = SuperLearner(scorer=f1, random_state=SEED)

# Build the first layer
layer1_classifiers = [MultinomialNB(),
                      SVC(probability=False),
                      LogisticRegression(),
                      XGBClassifier(n_estimators=500,
                                    max_depth=5,
                                    learning_rate=0.2,
                                    objective='binary:logistic',
                                    scale_pos_weight=2,
                                    n_jobs=-1,
                                    random_state=SEED),
                      RandomForestClassifier(random_state=SEED, n_estimators=500, n_jobs=-1)]
ensemble.add(layer1_classifiers)

# Attach the final meta estimator
ensemble.add_meta(RandomForestClassifier(random_state=SEED, n_estimators=500, n_jobs=-1))
pipeline = Pipeline([
    ('union', ColumnTransformer(
        [('cuisines_offered', CountVectorizer(min_df=10), 'cuisines_offered'),
         ('zipcode', OneHotEncoder(dtype='int', handle_unknown='ignore'), ['zipcode']),
         ('num_reviews', CountVectorizer(max_df=7, token_pattern='\d+'), 'num_reviews'),
         ('avg_rating', OneHotEncoder(dtype='int', handle_unknown='ignore'), ['avg_rating']),
         ('text', TfidfVectorizer(
             stop_words='english',
             strip_accents='unicode',
             min_df=3,
             max_df=0.5,
             ngram_range=(1, 3),
             max_features=500), 'preprocessed_texts')],
        remainder='passthrough',
    )),
    ('clf', ensemble)
], verbose=False)

def create_submission(y_pred, filepath):
    with open(filepath, 'w') as f:
        f.write('jc26\n')
        for label in y_pred:
            f.write(str(int(label)) + '\n')

# pipeline.fit(X_train, y_train)
# y_pred = pipeline.predict(X_test)
# score = metrics.f1_score(y_test, y_pred)
scores = cross_val_score(pipeline, train_preprocessed, train_labels, cv=5, scoring= 'f1')
print(scores)
print("Average F1-Score: %0.5f" % np.average(scores))

pipeline.fit(train, train_labels)
y_pred = pipeline.predict(test)
submit_path ='./submissions/submission7_mlens1.txt'
create_submission(y_pred, submit_path)

pipeline = Pipeline([
    ('preprocess', ColumnTransformer(
        [('cuisines_offered', CountVectorizer(min_df=10), 'cuisines_offered'),
         ('zipcode', OneHotEncoder(dtype='int', handle_unknown='ignore'), ['zipcode']),
         ('num_reviews', CountVectorizer(max_df=7, token_pattern='\d+'), 'num_reviews'),
         ('avg_rating', OneHotEncoder(dtype='int', handle_unknown='ignore'), ['avg_rating']),
         ('text', TfidfVectorizer(
             stop_words='english',
             strip_accents='unicode',
             min_df=3,
             max_df=0.5,
             ngram_range=(1, 3),
             max_features=500), 'preprocessed_texts')],
        remainder='passthrough',
    )),
    ('clf', MultinomialNB())
], verbose=False)

pipeline.fit(train_preprocessed, train_labels)
y_pred = pipeline.predict(test_preprocessed)
# score = metrics.f1_score(y_test, y_pred)
# scores = cross_val_score(pipeline, train_preprocessed, train_labels, cv=5, scoring= 'f1_macro')
# print(scores)
# print("Average F1-Score: %0.5f" % np.average(scores))

submit_path ='./submissions/submission10_SVM_preprocessed_countVectorizer.txt'
create_submission(y_pred, submit_path)

# https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer.html#sphx-glr-auto-examples-compose-plot-column-transformer-py
# https://towardsdatascience.com/nlp-performance-of-different-word-embeddings-on-text-classification-de648c6262b