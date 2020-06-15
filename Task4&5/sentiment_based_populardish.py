import datetime
import json
import pickle
import re

import chartify
import numpy as np
import metapy
import nltk
import pandas
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from utils import show_plots, write_to_file

path2files = "../data/"
path2buisness = path2files + "yelp_academic_dataset_business.json"
path2reviews = path2files + "yelp_academic_dataset_review.json"
outputfile = 'outputpath/flat_review_per_restaurant_with_rating.csv'


def process_restaurant_review_rating():
    rid2name = {}
    categories = set([])
    restaurant_ids = set([])
    cat2rid = {}
    rest2rate = {}
    rest2revID = {}
    r = 'Restaurants'
    target_cuisine = 'Indian'
    print('filtering for a single cuisine -> ' + target_cuisine)
    with open(path2buisness, 'r') as f:
        for line in f.readlines():
            business_json = json.loads(line)
            bjc = business_json['categories']
            # cities.add(business_json['city'])
            if r in bjc:
                if target_cuisine in bjc:
                    if len(bjc) > 1:
                        # print(bjc)
                        restaurant_ids.add(business_json['business_id'])
                        categories = set(bjc).union(categories) - set([r])
                        stars = business_json['stars']
                        rest2rate[business_json['business_id']] = stars
                        for cat in bjc:
                            if cat == r:
                                continue
                            if cat in cat2rid:
                                cat2rid[cat].append(business_json['business_id'])
                            else:
                                cat2rid[cat] = [business_json['business_id']]
                            rid2name[business_json['business_id']] = business_json['name']
    f.close()

    print('extraction done for restaurants and respective start rating for -> ' + target_cuisine)
    # important rest2rate, rid2name
    with open(path2reviews, 'r') as f:
        for line in f.readlines():
            review_json = json.loads(line)
            if review_json['business_id'] in restaurant_ids:
                if review_json['business_id'] in rest2revID:
                    rest2revID[review_json['business_id']].append(review_json['review_id'])
                else:
                    rest2revID[review_json['business_id']] = [review_json['review_id']]
    f.close()

    # important rest2revID
    print('extraction done for restaurants and respective review IDs -> ' + target_cuisine)

    print("reading from reviews file...")
    info_list = []
    with open(path2reviews, 'r') as f:
        for line in f.readlines():
            review_json = json.loads(line)
            rid = review_json['business_id']
            try:
                rname = rid2name[rid]
            except:
                rname = ''
            if rid in rest2rate:
                if review_json['text'] != '' or review_json['text'] is not None or rname != '' or rest2rate[
                    rid] != '' or rest2rate[
                    rid] is not None or review_json['stars'] != '' or review_json['stars'] is not None:
                    text = review_json['text'].replace('\n', '')
                    text = text.strip()
                    text = text.replace(',', '')
                    text = text.replace('"', '')
                    text = text.replace('\'', '')
                    text = text.replace('.', '')
                    info_list.append(
                        ','.join([rid, rname, str(int(rest2rate[rid])), text,
                                  str(review_json['stars'])]))
    f.close()
    with open(outputfile, 'w') as f:
        for item in info_list:
            f.write(item + '\n')
    print(
        'done writing csv file for “restaurant_id” , "restaurant_name", “restaurant_rating”, “review_text” and “review_rating”')


def document_features(document, word_feature):
    document_words = set(document)
    features = {}
    for w in word_feature:
        features['contains({})'.format(w)] = (w in document_words)
    return features


def build_collection(doc, filename):
    tok = metapy.analyzers.ICUTokenizer(suppress_tags=True)
    # tok = metapy.analyzers.LowercaseFilter(tok)
    tok = metapy.analyzers.ListFilter(tok, 'outputpath/' + filename,
                                      metapy.analyzers.ListFilter.Type.Reject)
    tok = metapy.analyzers.Porter2Filter(tok)
    tok.set_content(doc.content())
    return tok


def process_review(doc):
    tokens = []
    for d in doc:
        docu = metapy.index.Document()
        docu.content(d)
        tokens.append(build_collection(docu, 'stopwords.txt'))
    return tokens


# def sentiment_analysis_based_popular_dish(inputfile):
#     from nltk.classify.scikitlearn import SklearnClassifier
#     from sklearn.linear_model import LogisticRegression
#
#     print('begin classification')
#     collection = pandas.read_csv('outputpath/word_frequency_reviews.csv', names=['word', 'freq'])
#     word_features = list(collection['word'])[:5000]
#
#
#     column = ['restaurant_id', 'restaurant_rating', 'review_text', 'review_rating']
#     dataset = pandas.read_csv(inputfile, delimiter=',', names=column)
#
#     dataset = dataset[dataset['restaurant_id'].notnull()]
#     dataset = dataset[dataset['restaurant_rating'].notnull()]
#     dataset = dataset[dataset['review_text'].notnull()]
#     dataset = dataset[dataset['review_rating'].notnull()]
#
#     dataset['review_text'] = dataset['review_text'].str.lower()
#     dataset['restaurant_rating'] = dataset['restaurant_rating'].astype(int)
#     dataset['review_rating'] = dataset['review_rating'].astype(int)
#
#     from sklearn.utils import shuffle
#     dataset = shuffle(dataset)
#
#     print('Shape of the data set:', dataset.shape)
#     print(dataset.describe())
#     print(dataset.groupby('review_rating').size())
#     # show_plots(dataset, 'Purples_r', 'review_rating')
#
#     document_review = dataset['review_text']
#     document_sent = dataset['review_rating']
#
#     processed_review = process_review(document_review[:5000])
#     trainX = zip(processed_review, document_sent[:5000])
#     featuresets = [(document_features(review, word_features), sentiment) for (review, sentiment) in trainX]
#
#     train_set, test_set = featuresets[3000:], featuresets[:2000]
#
#     classifier = SklearnClassifier(LogisticRegression())
#     classifier.train(train_set)
#     print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(classifier, test_set))*100)
#     print('training model complete...')
#     test_dataframe = pandas.read_csv('outputpath/indian_dishnames_task3_freq_bigram.csv', names=['word', 'freq'])
#     content = ''
#     for ind, row in test_dataframe.iterrows():
#         word = row['word']
#         freq = row['freq']
#         p_review = process_review(word)
#         doc_test = set(p_review)
#         feat = {}
#         for w in word_features:
#             feat['contains({})'.format(w)] = (w in doc_test)
#         test = classifier.classify(feat)
#         content += u'{},{},{}\n'.format(word, freq, str(test))
#     write_to_file('outputpath', 'word_frequency_reviews_bigram_classify.csv', content)
#     print('classification done for input file')

def sentiment_analysis_based_popular_dish(inputfile):
    print('begin classification')

    column = ['restaurant_id', 'restaurant_name', 'restaurant_rating', 'review_text', 'review_rating']
    dataset = pandas.read_csv(inputfile, delimiter=',', names=column)

    dataset = dataset[dataset['restaurant_id'].notnull()]
    dataset = dataset[dataset['restaurant_rating'].notnull()]
    dataset = dataset[dataset['review_text'].notnull()]
    dataset = dataset[dataset['review_rating'].notnull()]

    dataset['review_text'] = dataset['review_text'].str.lower()
    dataset['restaurant_rating'] = dataset['restaurant_rating'].astype(int)
    dataset['review_rating'] = dataset['review_rating'].astype(int)

    from sklearn.utils import shuffle
    # df = shuffle(dataset)
    df = dataset
    stemmer = nltk.PorterStemmer()
    words = stopwords.words("english")
    df['cleaned'] = df['review_text'].apply(
        lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in words]).lower())

    vectorizer = TfidfVectorizer(min_df=1, stop_words="english", sublinear_tf=True, norm='l2', ngram_range=(1, 1))
    X = df['cleaned']
    Y = df['review_rating']
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=123)
    # instead of doing these steps one at a time, we can use a pipeline to complete them all at once
    pipeline = Pipeline([('vect', vectorizer),
                         ('chi', SelectKBest(chi2, k=1200)),
                         ('clf', RandomForestClassifier())])
    # fitting our model and save it in a pickle for later use
    model = pipeline.fit(X_train, y_train)
    with open('RandomForest.pickle', 'wb') as f:
        pickle.dump(model, f)
    ytest = np.array(y_test)
    print(classification_report(ytest, model.predict(X_test)))
    print(confusion_matrix(ytest, model.predict(X_test)))

    test_dataframe = pandas.read_csv('student_dn_annotations.txt', names=['word', 'freq'])
    abc = model.predict(test_dataframe['word'])

    # write_to_file('outputpath', 'word_frequency_reviews_bigram_classify.csv', 'content')
    print('classification done for input file')


def vader_sentiment_forPopular_restaurants():
    # for "chickken tikka"

    def sentiment_analyzer_scores(restname, sentence):
        score = analyser.polarity_scores(sentence)
        print("{} -> {}".format(restname, str(score)))
        return score

    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyser = SentimentIntensityAnalyzer()
    # sentiment_analyzer_scores("The phone is super cool.")
    column = ['restaurant_name', 'review_text', 'average_rating', 'number_of_reviews']
    dataset = pandas.read_csv('outputpath/rest_review_avg_rat.csv', delimiter=',', names=column)

    with open('outputpath/popular_dish_vader.csv', 'w') as f:
        for i, item in dataset.iterrows():
            scored = sentiment_analyzer_scores(item['restaurant_name'], item['review_text'])
            f.write('{},{},{},{},{},{}\n'.format(item['restaurant_name'], item['average_rating'], scored['neg'],
                                                 scored['neu'], scored['pos'], scored['compound']))

    f.close()


def restaurant_vader_sent_freq_to_graph():
    dataset = pandas.read_csv('outputpath/popular_dish_vader.csv',
                              names=['restaurant_name', 'avg_rating', 'neg', 'neu', 'pos', 'compound'])
    Y = dataset.sort_values('avg_rating')
    Y = Y[Y.avg_rating > 4.25]
    Y.head(len(Y))
    ch = chartify.Chart(blank_labels=True, x_axis_type='linear', y_axis_type='categorical', layout='slide_2000%')
    ch.set_title('Popular restaurant based on sentiments ("Chicken tikka")')
    ch.set_subtitle('By sentiment (color based on vader positive aspect sentiments [0 - 1])')
    ch.plot.bar(
        data_frame=Y,
        categorical_columns=['restaurant_name'],
        numeric_column='avg_rating',
        color_column='pos',
        categorical_order_ascending=True
    )

    ch.plot.text(
        data_frame=Y,
        categorical_columns=['restaurant_name'],
        numeric_column='avg_rating',
        text_column='avg_rating',
        color_column='pos',
        # font_size='1em',
    )

    ch.axes.set_xaxis_label('Average rating --->')
    ch.axes.set_yaxis_label('Restaurant Names --->')
    ch.style.set_color_palette('categorical', 'Dark2')
    ch.axes.set_xaxis_tick_orientation('horizontal')
    ch.axes.set_yaxis_tick_orientation('horizontal')
    ch.set_legend_location(None)
    ch.show()


def dish_sent_freq_to_graph():
    # this is used after sentiment.py
    # removed few repetitive  data
    dataset = pandas.read_csv('outputpath/dish_senti.csv', names=['dish', 'freq', 'senti'])
    Y = dataset.sort_values('freq')
    Y = Y[Y.freq > 20]
    Y.head(len(Y))
    ch = chartify.Chart(blank_labels=True, x_axis_type='linear', y_axis_type='categorical', layout='slide_2000%')
    ch.set_title("Popular Dishes based on sentiments")
    ch.set_subtitle('By sentiment (color based on sentiments [1 - 5])')
    ch.plot.bar(
        data_frame=Y,
        categorical_columns=['dish'],
        numeric_column='freq',
        color_column='senti',
        categorical_order_ascending=True
    )

    ch.plot.text(
        data_frame=Y,
        categorical_columns=['dish'],
        numeric_column='freq',
        text_column='freq',
        color_column='senti',
        # font_size='1em',
    )

    ch.axes.set_xaxis_label('Frequency --->')
    ch.axes.set_yaxis_label('Dish Names --->')
    ch.style.set_color_palette('categorical', 'Dark2')
    ch.axes.set_xaxis_tick_orientation('horizontal')
    ch.axes.set_yaxis_tick_orientation('horizontal')
    ch.set_legend_location(None)
    ch.show()


st_time = datetime.datetime.now()
# process_restaurant_review_rating()
# sentiment_analysis_based_popular_dish(outputfile)
# dish_sent_freq_to_graph()
# vader_sentiment_forPopular_restaurants()
restaurant_vader_sent_freq_to_graph()
en_time = datetime.datetime.now()
print('Total execution time (milliseconds): ' + str((en_time - st_time).total_seconds() * 1000))
