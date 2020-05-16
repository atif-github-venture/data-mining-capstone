import datetime
import json
import os
from collections import Counter
from plotly.graph_objs import *
import pandas
from plotly.graph_objs import graph_objs
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.ldamodel import LdaModel
from gensim import matutils
from matplotlib import pyplot as plt
from wordcloud import WordCloud
import matplotlib.colors as mcolors
from utils import read_json_content


def normalize(values):
    total = 0.0
    for item in values:
        for k, v in item.items():
            total += float(v)
    return [total * 1000 / len(values)]


def build_text(key, values):
    result = 'Topic ' + str(key) + ':<br>'

    for item in values:
        for k, v in item.items():
            result += (k + ':' + v + '<br>')

    return [result]


def build_layout():
    layout = Layout(
        title='Cluster of Topic0 - Topic10 by LDP',
        showlegend=False,
        height=600,
        width=800,
        xaxis=graph_objs.layout.XAxis(
            title='Topic ID',
            gridcolor='rgb(255, 255, 255)',
            zerolinewidth=1,
            ticklen=5,
            gridwidth=1,
        ),
        yaxis=graph_objs.layout.YAxis(
            title='Group Frequency (1000x)',
            gridcolor='rgb(255, 255, 255)',
            zerolinewidth=1,
            ticklen=5,
            gridwidth=2,
        ),
        paper_bgcolor='rgb(243, 243, 243)',
        plot_bgcolor='rgb(243, 243, 243)',
    )
    return layout


def display(dataset):
    for k in dataset:
        print('\nkey: ', k)
        print('\nvalues: ', dataset[k])


def build_trace(dataset):
    traces = []
    for key in sorted(dataset.keys()):
        trace = 'trace' + str(key)
        trace = Scatter(
            x=[key],
            y=normalize(dataset[key]),
            text=build_text(key, dataset[key]),
            mode='markers',
            name=key,
            marker=graph_objs.scatter.Marker(
                sizemode='diameter',
                sizeref=0.85,
                size=normalize(dataset[key]),
                # line=graph_objs.scatter.Line(),
            )
        )

        traces.append(trace)

    return traces


class YelpReviewsLDA:
    def __init__(self, input_file, output_file, num_of_topics, num_of_feature, words_per_topic):
        self.path = os.path.join(os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)), os.pardir)),
                                 'data')
        self.output_path = os.path.join(
            os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)), os.pardir)),
            'output')
        self.text = None
        self.input_file = input_file
        self.output_file = output_file
        self.num_of_topics = num_of_topics  # Number of topics to use when running the LDA algorithm
        self.num_of_features = num_of_feature  # Feature is the number of features to keep when mapping the bag-of-words
        # to tf-idf vectors, (eg. length of vectors)
        self.words_per_topic = words_per_topic  # This option specifies how many words to display for each topic

    def lda(self, mode=None):
        K_clusters = self.num_of_topics
        vectorizer = TfidfVectorizer(max_df=0.5, max_features=self.num_of_features,
                                     min_df=2, stop_words='english',
                                     use_idf=True)

        dataset = pandas.read_csv(self.path + '/' + self.input_file)
        dataset = dataset[dataset['text'].notnull()]
        if mode == 'positive':
            dataset = dataset[dataset['stars'] >= 3]
        elif mode == 'negative':
            dataset = dataset[dataset['stars'] <= 2]
        self.text = dataset['text']
        # self.text = self.text[:5000]

        print("Extracting features from the training dataset using a sparse vectorizer")
        X = vectorizer.fit_transform(self.text)
        print("n_samples: %d, n_features: %d" % X.shape)

        # Mapping from feature id to actual word
        id2words = {}
        for i, word in enumerate(vectorizer.get_feature_names()):
            id2words[i] = word

        print("Applying topic modeling, using LDA")
        print(str(K_clusters) + " topics")
        corpus = matutils.Sparse2Corpus(X, documents_columns=False)
        lda = LdaModel(corpus, num_topics=K_clusters, id2word=id2words)
        topics = lda.show_topics(num_topics=K_clusters, num_words=self.words_per_topic, formatted=False)
        output_dict = {}
        for i, item in enumerate(topics):
            child = []
            for term, weight in item[1]:
                child.append({term: str(weight)})
            output_dict[item[0]] = child
        print(output_dict)
        with open(self.output_path + '/' + self.output_file, 'w') as fp:
            json.dump(output_dict, fp)
        fp.close()
        self.generate_word_cloud(topics)
        self.word_count_per_topic(topics)

    def generate_word_cloud(self, topics):
        cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
        cloud = WordCloud(background_color='white',
                          width=2500,
                          height=1800,
                          max_words=10,
                          colormap='tab10',
                          prefer_horizontal=1.0)
        # NOTE: Change the size of graph matrix based on number of topics
        fig, axes = plt.subplots(5, 2, figsize=(10, 10), sharex=True, sharey=True)

        for i, ax in enumerate(axes.flatten()):
            fig.add_subplot(ax)
            topic_words = dict(topics[i][1])
            cloud.generate_from_frequencies(topic_words, max_font_size=300)
            plt.gca().imshow(cloud)
            plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
            plt.gca().axis('off')
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.axis('off')
        plt.margins(x=0, y=0)
        plt.tight_layout()
        # plt.show()
        plt.savefig(self.output_path + '/' + 'generate_word_cloud.png')

    def visualization(self):
        dataset = read_json_content(self.output_path + '/' + self.output_file)
        traces = build_trace(dataset)
        print(traces)
        layout = build_layout()
        fig = Figure(data=traces, layout=layout)
        fig.show(filename='task1.1 LDP Topic Sample')

    def word_count_per_topic(self, topics):
        data_flat = [w for w_list in self.text for w in w_list]
        counter = Counter(data_flat)

        out = []
        for i, topic in topics:
            for word, weight in topic:
                out.append([word, i, weight, counter[word]])

        df = pandas.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])

        # Plot Word Count and Weights of Topic Keywords
        fig, axes = plt.subplots(5, 2, figsize=(16, 10), sharey=True, dpi=160)
        cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
        for i, ax in enumerate(axes.flatten()):
            ax.bar(x='word', height="word_count", data=df.loc[df.topic_id == i, :], color=cols[i], width=0.5, alpha=0.3,
                   label='Word Count')
            ax_twin = ax.twinx()
            ax_twin.bar(x='word', height="importance", data=df.loc[df.topic_id == i, :], color=cols[i], width=0.2,
                        label='Weights')
            ax.set_ylabel('Word Count', color=cols[i])
            ax_twin.set_ylim(0, 0.030)
            ax.set_ylim(0, 3500)
            ax.set_title('Topic: ' + str(i), color=cols[i], fontsize=16)
            ax.tick_params(axis='y', left=False)
            ax.set_xticklabels(df.loc[df.topic_id == i, 'word'], rotation=30, horizontalalignment='right')
            ax.legend(loc='upper left')
            ax_twin.legend(loc='upper right')

        fig.tight_layout(w_pad=2)
        fig.suptitle('Word Count and Importance of Topic Keywords', fontsize=22, y=1.05)
        # plt.show()
        plt.savefig(self.output_path + '/' + 'word_count_per_topic.png')


def main():
    st_time = datetime.datetime.now()
    input_file = 'yelp_academic_dataset_review.csv'
    output_file = 'yelp_academic_dataset_review_topic.txt'

    ra = YelpReviewsLDA(input_file, output_file, 10, 50000, 15)
    # Task 1.1 - for all reviews
    ra.lda()
    # Task 1.2 - uncomment individually for positive/negative, but make sure to comment ra.lda() when using for this.
    # ra.lda(mode='positive')
    # ra.lda(mode='negative')
    # For visualization of all the reviews
    ra.visualization()
    en_time = datetime.datetime.now()
    print('Total execution time (milliseconds): ' + str((en_time - st_time).total_seconds() * 1000))


if __name__ == "__main__":
    main()
