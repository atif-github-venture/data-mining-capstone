import datetime
import chartify
import metapy
import pandas
import seaborn as sns
import matplotlib.pyplot as plt

path2files = "../data/"
path2buisness = path2files + "yelp_academic_dataset_business.json"
path2reviews = path2files + "yelp_academic_dataset_review.json"
outputfile = 'outputpath/flat_review_per_restaurant_with_rating.csv'


def review_text_per_restaurant(inputfile):
    print('begin review_text_per_restaurant')
    column = ['restaurant_id', 'restaurant_name', 'restaurant_rating', 'review_text', 'review_rating']
    dataset = pandas.read_csv(inputfile, delimiter=',', names=column)
    dataset = dataset[dataset['restaurant_id'].notnull()]
    dataset = dataset[dataset['restaurant_rating'].notnull()]
    dataset = dataset[dataset['review_text'].notnull()]
    dataset = dataset[dataset['review_rating'].notnull()]
    dataset['review_text'] = dataset['review_text'].str.lower()
    dataset['restaurant_rating'] = dataset['restaurant_rating'].astype(int)
    dataset['review_rating'] = dataset['review_rating'].astype(int)

    # get unique list of restaurant names
    dish_name = 'chicken tikka'
    aggregated_reviews = []
    rest_names = set(dataset['restaurant_name'])
    for item in rest_names:
        filter = dataset['restaurant_name'] == item
        query_set = dataset.where(filter, inplace=False)
        query_set = query_set.dropna(subset=['review_text'])
        review_set = ''
        count = 0
        size = 0
        for i, row in query_set.iterrows():
            if dish_name in row['review_text']:
                review_set = review_set + ' ' + row['review_text']
                count = count + (row['review_rating'])
                size += 1
        text = review_set.replace('\n', '')
        text = text.strip()
        text = text.replace(',', '')
        text = text.replace('"', '')
        text = text.replace('\'', '')
        text = text.replace('.', '')
        if text != '' and size != 0:
            aggregated_reviews.append(tuple((item, text, count / size, size)))

    with open('outputpath/rest_review_avg_rat.csv', 'w') as f:
        for x, y, z, c in aggregated_reviews:
            if x == 'Mint Indian Bistro':
                print('Mint Indian Bistro')
            f.write(u'{},{},{},{}\n'.format(x, y, z, c))

    column = ['restaurant_name', 'review_text', 'avg_rating', 'number_of_reviews']
    dataset = pandas.read_csv('outputpath/rest_review_avg_rat.csv', delimiter=',', names=column)
    Y = dataset.sort_values('avg_rating')
    Y = Y[Y.avg_rating > 4]
    Y.head(len(Y))
    ch = chartify.Chart(blank_labels=True, x_axis_type='linear', y_axis_type='categorical', layout='slide_2000%')
    ch.set_title("Popular Restaurants - Chicken Tikka")
    ch.set_subtitle('By average rating (color for distinction ONLY)')
    ch.plot.bar(
        data_frame=Y,
        categorical_columns=['restaurant_name'],
        numeric_column='avg_rating',
        color_column='restaurant_name',
        categorical_order_ascending=True
    )

    ch.plot.text(
        data_frame=Y,
        categorical_columns=['restaurant_name'],
        numeric_column='avg_rating',
        text_column='avg_rating',
        color_column='restaurant_name',
        # font_size='1em',
    )

    ch.axes.set_xaxis_label('Average rating based on reviews --->')
    ch.axes.set_yaxis_label('Restaurant Names --->')
    ch.style.set_color_palette('categorical', 'Dark2')
    ch.axes.set_xaxis_tick_orientation('horizontal')
    ch.axes.set_yaxis_tick_orientation('horizontal')
    ch.set_legend_location(None)
    ch.show()

    print('end review_text_per_restaurant')


def search_dish_name_restaurant_review():
    print('begin search_dish_name_restaurant_review')
    idx = metapy.index.make_inverted_index('config.toml')
    print(idx.num_docs())
    ranker = metapy.index.OkapiBM25()
    query = metapy.index.Document()
    query.content('chicken tikka')
    top_docs = ranker.score(idx, query, num_results=100)
    print(top_docs)

    with open('outputpath/search_ranking_chicken_tikka.csv', 'w') as f:
        for item in top_docs:
            f.write(str(item[0])+','+str(item[1])+'\n')
    f.close()
    # for num, (d_id, _) in enumerate(top_docs):
    #     print(idx.metadata(d_id).get('content'))
    print('*****************')
    print('end search_dish_name_restaurant_review')


def plot_per_search(inputfile):
    print('begin plot_per_search')
    column = ['restaurant_name', 'review_text', 'avg_rating', 'number_of_reviews']
    dataset = pandas.read_csv(inputfile, delimiter=',', names=column)

    rest_name = []
    rest_score = []
    column = ['document_index', 'score']
    documentset = pandas.read_csv('outputpath/search_ranking_chicken_tikka.csv', delimiter=',', names=column)
    for i , item in documentset.iterrows():
        search = item['document_index']
        for i , dataset_item in dataset.iterrows():
            if i == int(search):
                rest_name.append(dataset_item['restaurant_name'])
                rest_score.append(item['score'])

    with open('outputpath/search_ranking_chicken_tikka_restname.csv', 'w') as f:
        for i in range(len(rest_name)):
            f.write('{},{}\n'.format(rest_name[i], rest_score[i]))
    f.close()

    column = ['restaurant_name', 'score']
    Y = pandas.read_csv('outputpath/search_ranking_chicken_tikka_restname.csv', delimiter=',', names=column)
    Y['score'] = Y['score'].astype(float)
    Y = Y.sort_values('score')
    Y = Y[Y.score > 1]
    Y.head(len(Y))
    sns.barplot(x="restaurant_name", y="score", data=Y, alpha=0.8)
    plt.xticks(rotation=65, horizontalalignment='right')
    plt.title('Ranking by document scoring algorithm', fontsize=18)
    plt.xlabel('Restaurant names\n(for "chicken tikka" dish)', fontsize=14)
    plt.ylabel('Document scores', fontsize=14)
    plt.show()

    print('end plot_per_search')






    print('end plot_per_search')

st_time = datetime.datetime.now()
# review_text_per_restaurant(outputfile)
# search_dish_name_restaurant_review()
plot_per_search('outputpath/rest_review_avg_rat.csv')
en_time = datetime.datetime.now()
print('Total execution time (milliseconds): ' + str((en_time - st_time).total_seconds() * 1000))
