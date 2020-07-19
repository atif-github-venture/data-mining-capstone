import datetime

import gensim
import numpy as np
import metapy
import pandas
from nltk.corpus import stopwords
from sklearn.utils.tests.test_pprint import CountVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.compose import ColumnTransformer
from gensim.parsing.preprocessing import strip_tags, strip_punctuation, strip_numeric, strip_short
from gensim.parsing.preprocessing import strip_multiple_whitespaces, strip_non_alphanum, remove_stopwords, stem_text
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn import svm
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from tqdm import tqdm
from utils import show_plots

SEED = 26
FILTERS_LIST = [lambda x: x.lower(),  # lowercase
                strip_tags,  # remove tags
                strip_punctuation,  # replace punctuation characters with spaces
                strip_multiple_whitespaces,  # remove repeating whitespaces
                # strip_numeric, # remove numbers
                gensim.parsing.preprocessing.remove_stopwords,  # remove stopwords
                strip_short,  # remove words less than minsize=3 characters long]
                stem_text]
def create_submission(y_pred, filepath):
    with open(filepath, 'w') as f:
        f.write('atifa2\n')
        for label in y_pred:
            f.write(str(int(label)) + '\n')

def build_collection(doc, filename):
    tok = metapy.analyzers.ICUTokenizer(suppress_tags=True)
    tok = metapy.analyzers.LowercaseFilter(tok)
    tok = metapy.analyzers.ListFilter(tok, filename,
                                      metapy.analyzers.ListFilter.Type.Reject)
    tok = metapy.analyzers.Porter2Filter(tok)
    tok.set_content(doc.content())
    return tok


def process_review(doc):
    tokens = []
    for d in doc:
        docu = metapy.index.Document()
        docu.content(d)
        tok = build_collection(docu, 'stopwords.txt')
        tokens.append(tok)
    return tokens


def preprocess(text):
    """
    strip_tags, strip_punctuation, strip_multiple_whitespaces, strip_numeric,
    """
    result_stemmed = []
    for token in gensim.parsing.preprocessing.preprocess_string(text, FILTERS_LIST):
        result_stemmed.append(WordNetLemmatizer().lemmatize(token))
    return result_stemmed


def pre_processing(text_path, labels_path, additional_path):
    print('pre-processing starting...')
    preprocessed_texts = []

    with open(text_path) as f:
        texts = f.readlines()

    for _text in tqdm(texts):
        result_stemmed = preprocess(_text)
        preprocessed_texts.append(result_stemmed)

    all_preprocessed_texts = [" ".join(_text) for _text in preprocessed_texts]
    with open(labels_path, 'r') as f:
        labels = [l.rstrip() for l in f]
    df = pandas.DataFrame({"label": labels, "text": texts,
                           "preprocessed_texts": all_preprocessed_texts,
                           "tokenized_texts": preprocessed_texts})
    hygiene_additional = pandas.read_csv(additional_path,
                                         names=["cuisines_offered", "zipcode", "num_reviews", "avg_rating"],
                                         dtype={"cuisines_offered": str,
                                                "zipcode": str,
                                                "num_reviews": str})
    df = df.join(hygiene_additional)
    df['avg_rating'] = df['avg_rating'].apply(lambda x: str(int(round(x, 0))))
    print('pre-processing complete...')
    return df


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
    scores = cross_val_score(pipeline, X, y, cv=5, scoring='f1_macro')
    print(clf)
    print(scores)
    cv_score = np.average(scores)
    return cv_score, pipeline


def classification_task(df):
    print('Begin classification_task()...')
    train_df = df[df["label"] != "[None]"]
    test_df = df[df["label"] == "[None]"]
    additional_feats = ["cuisines_offered", "zipcode", "num_reviews", "avg_rating"]
    train = train_df[["text"] + additional_feats]
    train_preprocessed = train_df[["preprocessed_texts"] + additional_feats]
    train_tokenized = train_df[["tokenized_texts"] + additional_feats]
    train_labels = train_df["label"].astype(int)  # needed by sklearn
    test = test_df[["text"] + additional_feats]
    test_preprocessed = test_df[["preprocessed_texts"] + additional_feats]
    test_tokenized = test_df[["tokenized_texts"] + additional_feats]
    test_labels = test_df["label"]

    print(train.shape, train_preprocessed.shape, train_tokenized.shape, train_labels.shape)
    print(test.shape, test_preprocessed.shape, test_tokenized.shape, test_labels.shape)
    classifiers = {
        'Naive Bayes': MultinomialNB(),
        # 'Support Vector Machine': svm.SVC(),
        # 'Logistic Regression': LogisticRegression(),
        # 'Random Forest': RandomForestClassifier(random_state=SEED, n_estimators=500, n_jobs=-1),
    }

    tfidf = TfidfVectorizer(
        stop_words='english',
        strip_accents='unicode',
        min_df=3,
        max_df=0.5,
        ngram_range=(1, 3),
        max_features=500)
    bow = CountVectorizer(
        stop_words=set(stopwords.words('english')),
        strip_accents='unicode',
        min_df=15,
        max_df=0.5,
        ngram_range=(1, 3))

    # # BOW - No Preprocessing
    # for clf_name, clf in classifiers.items():
    #     cv_score = test_classifier(clf, train, train_labels,
    #                                vectorizer=bow, text_col='text')
    #     print('{}: {}'.format(clf_name, cv_score))
    #
    # # BOW - Preprocessing
    # for clf_name, clf in classifiers.items():
    #     cv_score = test_classifier(clf, train_preprocessed, train_labels,
    #                                vectorizer=bow, text_col='preprocessed_texts')
    #     print('{}: {}'.format(clf_name, cv_score))
    #
    # # TFIDF - No Preprocessing
    # for clf_name, clf in classifiers.items():
    #     cv_score = test_classifier(clf, train, train_labels,
    #                                vectorizer=tfidf, text_col='text')
    #     print('{}: {}'.format(clf_name, cv_score))

    # TFIDF - Preprocessing

    for clf_name, clf in classifiers.items():
        cv_score, pipeli = test_classifier(clf, train_preprocessed, train_labels,
                                   vectorizer=tfidf, text_col='preprocessed_texts')
        print('{}: {}'.format(clf_name, cv_score))
        pipeli.fit(train_preprocessed, train_labels)
        y_pred = pipeli.predict(test_preprocessed)
        submit_path ='output_'+clf_name+'.txt'
        create_submission(y_pred, submit_path)
    print('End classification_task()...')


hygiene_text_path = "./Hygiene/hygiene.dat"
hygiene_labels_path = "./Hygiene/hygiene.dat.labels"
hygiene_additional_path = "./Hygiene/hygiene.dat.additional"

st_time = datetime.datetime.now()
dataframe = pre_processing(hygiene_text_path, hygiene_labels_path, hygiene_additional_path)
classification_task(dataframe)
en_time = datetime.datetime.now()
print('Total execution time (milliseconds): ' + str((en_time - st_time).total_seconds() * 1000))
