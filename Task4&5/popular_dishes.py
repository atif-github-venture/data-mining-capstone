import sys
import gensim
import chartify
import logging
import matplotlib.pyplot as plt
import metapy
import nltk
import pandas
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import seaborn as sns

from utils import write_to_file


def build_collection(doc, filename):
    tok = metapy.analyzers.ICUTokenizer(suppress_tags=True)
    tok = metapy.analyzers.LengthFilter(tok, min=1, max=3)
    tok = metapy.analyzers.LowercaseFilter(tok)
    tok = metapy.analyzers.ListFilter(tok, 'outputpath/' + filename,
                                      metapy.analyzers.ListFilter.Type.Reject)
    tok = metapy.analyzers.Porter2Filter(tok)
    tok.set_content(doc.content())
    return tok


def word_frequency(input_file, output, output_file):
    """
    Collect the words from dataset text column, count the frequency of each word (excluding stop words)
    Save to file
    :return: None
    """
    all_text = ''
    with open(input_file) as fh:
        for line in fh:
            line = line.strip()
            all_text = all_text + line

    nltk.bigrams(all_text)
    doc = metapy.index.Document()
    doc.content(all_text)
    tokens = build_collection(doc, 'stopwords.txt')

    all_words = nltk.FreqDist(tokens)
    string = ''
    for word, frequency in all_words.most_common():
        string += u'{},{}\n'.format(word, frequency)

    write_to_file(output, output_file, string)

    df = pandas.DataFrame(all_words.most_common())
    df.columns = ['Word', 'Freq']
    print(df)
    ax = df.plot(legend=True, title='Word frequency distribution (Zipf\'s law)')
    ax.set_xlabel('Words', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    plt.show()


def look_up(search, search_db, output_file):
    # convert search word per line into an array
    search_array = []
    with open(search) as fh:
        for line in fh:
            line = line.strip()
            search_array.append(line)

    column = ['word', 'freq']
    X = pandas.read_csv(search_db, names=column, delimiter=',')
    Y = X[X['word'].isin(search_array)]
    Y.to_csv(output_file, index=False, header=False)
    Y['freq'] = Y['freq'].astype(int)

    ax = Y.plot(legend=True, title='Popular dish names occurrence')
    ax.set_xlabel('Dish names', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    plt.show()

    Y = Y[Y.freq > 25]
    Y = Y.sort_values('freq')
    Y.head(len(Y))
    ch = chartify.Chart(blank_labels=True, x_axis_type='linear', y_axis_type='categorical', layout='slide_2000%')
    ch.set_title("Popular Dishes - Indian Cuisine")
    ch.set_subtitle('By number of occurrences in review text (color for distinction ONLY)')
    ch.plot.bar(
        data_frame=Y,
        categorical_columns=['word'],
        numeric_column='freq',
        color_column='word',
        categorical_order_ascending=True
    )

    ch.plot.text(
        data_frame=Y,
        categorical_columns=['word'],
        numeric_column='freq',
        text_column='freq',
        color_column='word',
        # font_size='1em',
    )

    ch.axes.set_xaxis_label('Frequency (# of occurrences) --->')
    ch.axes.set_yaxis_label('Dish Names --->')
    ch.style.set_color_palette('categorical', 'Dark2')
    ch.axes.set_xaxis_tick_orientation('horizontal')
    ch.axes.set_yaxis_tick_orientation('horizontal')
    ch.set_legend_location(None)
    ch.show()


def word_frequency_bigram(input_file, output, output_file):
    f = open(input_file)
    raw = f.read()
    f.close()

    stopwords = []
    with open('outputpath/stopwords.txt') as fh:
        for line in fh:
            line = line.strip()
            stopwords.append(line)

    tokens = nltk.word_tokenize(raw.lower())

    wordsFiltered = []
    for w in tokens:
        if w not in stopwords:
            wordsFiltered.append(w)

    # Create your bigrams
    bgs = nltk.bigrams(wordsFiltered)

    # compute frequency distribution for all the bigrams in the text
    fdist = nltk.FreqDist(bgs)
    string = ''
    for word, frequency in fdist.most_common():
        string += u'{},{}\n'.format(' '.join(word), int(frequency))

    write_to_file(output, output_file, string)
    print('done: word_frequency_bigram')


if __name__ == '__main__':
    method = 'lookup_and_visualize_bigram'

    if method == 'word_frequency_unigram':
        input_filename = 'Indian_comments.txt'
        output_path = 'outputpath'
        output_filename = 'word_frequency_reviews.csv'
        word_frequency(input_filename, output_path, output_filename)
    elif method == 'word_frequency_bigram':
        input_filename = 'Indian_comments.txt'
        output_path = 'outputpath'
        output_filename = 'word_frequency_reviews_bigram.csv'
        word_frequency_bigram(input_filename, output_path, output_filename)
    elif method == 'lookup_and_visualize_unigram':
        # search_input = 'student_dn_annotations.txt'
        # output_filename = 'outputpath/student_dn_annotations_freq.csv'
        search_input = 'indian_dishnames_task3.txt'
        output_filename = 'outputpath/indian_dishnames_task3_freq.csv'
        search_database = 'outputpath/word_frequency_reviews.csv'
        look_up(search_input, search_database, output_filename)
    elif method == 'lookup_and_visualize_bigram':
        # search_input = 'student_dn_annotations.txt'
        # output_filename = 'outputpath/student_dn_annotations_freq_bigram.csv'
        search_input = 'indian_dishnames_task3.txt'
        output_filename = 'outputpath/indian_dishnames_task3_freq_bigram.csv'
        search_database = 'outputpath/word_frequency_reviews_bigram.csv'
        look_up(search_input, search_database, output_filename)
