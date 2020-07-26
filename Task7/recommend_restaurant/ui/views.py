from bokeh.embed import file_html
from django.shortcuts import render
import chartify
import pandas
from django.contrib import messages


def home(request):
    plot_div = ""
    searchtype = None
    vader_result = None
    addr = None
    dataset = pandas.read_csv('ui/' + 'indian_dishnames.csv', delimiter=',', names=['dishnames', 'freq'])
    dishnames = dataset['dishnames']
    if request.method == 'GET':
        try:
            searchtype = request.GET['searchtype']
        except:
            pass
    elif request.method == 'POST':
        searchtype = request.GET['searchtype']
        if 'searchtype1' == request.GET['searchtype']:
            dishname = request.POST['dishname']
            rating = request.POST['customRange']
            size = request.POST['result_size']
            if dishname == 'Select a dish':
                dataset = pandas.read_csv('ui/indian_dishnames.csv', delimiter=',', names=['dishnames', 'freq'])
                dishnames = dataset['dishnames']
                messages.info(request, "Please select a dish name.", '')
                return render(request, 'ui/home.html',
                              {'plot_div': plot_div, 'dishnames': dishnames, 'searchtype': searchtype})
            else:
                plot_div, addr = review_text_per_restaurant(dishname, rating, int(size))
        elif 'searchtype2' == request.GET['searchtype']:
            dishname = request.POST['dishname']
            rating = request.POST['customRange']
            size = request.POST['result_size']
            if dishname == 'Select a dish':
                dataset = pandas.read_csv('ui/' + 'indian_dishnames.csv', delimiter=',', names=['dishnames', 'freq'])
                dishnames = dataset['dishnames']
                messages.info(request, "Please select a dish name.", '')
                return render(request, 'ui/home.html',
                              {'plot_div': plot_div, 'dishnames': dishnames, 'searchtype': searchtype})
            else:
                plot_div, vader_result, addr = vader_sentiment_forPopular_restaurants(dishname, rating, int(size))
    return render(request, 'ui/home.html', {'plot_div': plot_div, 'dishnames': dishnames, 'searchtype': searchtype,
                                            'vader_result': vader_result, 'addr': addr})


def extract_based_on_dish_name(dish):
    inputfile = 'flat_review_per_restaurant_with_rating.csv'
    column = ['restaurant_id', 'restaurant_name', 'restaurant_rating', 'review_text', 'review_rating', 'address']
    dataset = pandas.read_csv('ui/' + inputfile, delimiter=',', names=column)
    dataset = dataset[dataset['restaurant_id'].notnull()]
    dataset = dataset[dataset['restaurant_rating'].notnull()]
    dataset = dataset[dataset['review_text'].notnull()]
    dataset = dataset[dataset['review_rating'].notnull()]
    dataset['review_text'] = dataset['review_text'].str.lower()
    dataset['restaurant_rating'] = dataset['restaurant_rating'].astype(int)
    dataset['review_rating'] = dataset['review_rating'].astype(int)

    # get unique list of restaurant names
    aggregated_reviews = []
    rest_names = set(dataset['restaurant_name'])
    for item in rest_names:
        filter = dataset['restaurant_name'] == item
        query_set = dataset.where(filter, inplace=False)
        query_set = query_set.dropna(subset=['review_text'])
        review_set = ''
        count = 0
        size = 0
        addr = ''
        for i, row in query_set.iterrows():
            if dish in row['review_text']:
                review_set = review_set + ' ' + row['review_text']
                count = count + (row['review_rating'])
                size += 1
            addr = row['address']
        text = review_set.replace('\n', '')
        text = text.strip()
        text = text.replace(',', '')
        text = text.replace('"', '')
        text = text.replace('\'', '')
        text = text.replace('.', '')
        if text != '' and size != 0:
            aggregated_reviews.append(tuple((item, text, count / size, size, addr)))

    with open('ui/' + 'rest_review_avg_rat.csv', 'w') as f:
        for x, y, z, c, a in aggregated_reviews:
            f.write(u'{},{},{},{},{}\n'.format(x, y, z, c, a))


def review_text_per_restaurant(dish_name, avg_rating, result_size):
    extract_based_on_dish_name(dish_name)
    column = ['restaurant_name', 'review_text', 'avg_rating', 'number_of_reviews', 'address']
    dataset = pandas.read_csv('ui/' + 'rest_review_avg_rat.csv', delimiter=',', names=column)
    Y = dataset.sort_values('avg_rating')
    Y = Y[Y.avg_rating >= float(avg_rating)]
    if len(Y) > result_size:
        Y = Y[:result_size]
    Y.head(len(Y))
    addr = []
    for i, item in Y.iterrows():
        addr.append(tuple((item['restaurant_name'], item['address'])))
    ch = chartify.Chart(blank_labels=True, x_axis_type='linear', y_axis_type='categorical', layout='slide_2000%')
    ch.set_title("Popular Restaurants - " + dish_name)
    ch.set_subtitle('By average rating low to high - top down (color for distinction ONLY)')
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
    output_html = file_html(ch.figure, 'cdn')
    return output_html, addr


def vader_sentiment_forPopular_restaurants(dish_name, avg_rating, result_size):
    extract_based_on_dish_name(dish_name)

    def sentiment_analyzer_scores(restname, sentence):
        score = analyser.polarity_scores(sentence)
        # print("{} -> {}".format(restname, str(score)))
        return score

    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyser = SentimentIntensityAnalyzer()
    # sentiment_analyzer_scores("The phone is super cool.")
    column = ['restaurant_name', 'review_text', 'average_rating', 'number_of_reviews', 'address']
    dataset = pandas.read_csv('ui/rest_review_avg_rat.csv', delimiter=',', names=column)
    addr = []
    with open('ui/popular_dish_vader.csv', 'w') as f:
        for i, item in dataset.iterrows():
            scored = sentiment_analyzer_scores(item['restaurant_name'], item['review_text'])
            addr.append(tuple((item['restaurant_name'] , item['address'])))
            f.write('{},{},{},{},{},{}\n'.format(item['restaurant_name'], item['average_rating'], scored['neg'],
                                                 scored['neu'], scored['pos'], scored['compound']))
    f.close()
    plot, data = restaurant_vader_sent_freq_to_graph(dish_name, avg_rating, result_size)
    res_dat = []
    addr1=[]
    for i, item in data.iterrows():
        a = []
        r_name = item['restaurant_name']
        a.append(r_name)
        a.append(item['neg'])
        a.append(item['neu'])
        a.append(item['pos'])
        a.append(item['compound'])
        res_dat.append(a)
        addr1.append([item for item in addr if item[0].startswith(r_name)][0])
    return plot, res_dat, addr1


def restaurant_vader_sent_freq_to_graph(dishname, avgrating, resultsize):
    dataset = pandas.read_csv('ui/popular_dish_vader.csv',
                              names=['restaurant_name', 'avg_rating', 'neg', 'neu', 'pos', 'compound'])
    Y = dataset.sort_values('avg_rating')
    Y = Y[Y.avg_rating >= float(avgrating)]
    if len(Y) > resultsize:
        Y = Y[:resultsize]
    Y.head(len(Y))
    ch = chartify.Chart(blank_labels=True, x_axis_type='linear', y_axis_type='categorical', layout='slide_2000%')
    ch.set_title('Popular restaurant based on sentiments ("' + dishname + '")')
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
    output_html = file_html(ch.figure, 'cdn')
    return output_html, Y
