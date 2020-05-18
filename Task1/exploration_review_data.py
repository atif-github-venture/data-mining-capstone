import csv
import datetime
import os
import nltk
import pandas
import matplotlib.pyplot as plt
import metapy

from utils import read_text, convert_string_to_json, write_to_txt, write_to_file, show_plots


class YelpReviewsAnalysis:
    def __init__(self, input_file, output_file, header):
        self.path = os.path.join(os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)), os.pardir)),
                                 'data')
        self.dir_path = os.path.join(
            os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)), os.pardir)),
            )
        self.output_path = os.path.join(
            os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)), os.pardir)),
            'output')
        self.input_file = input_file
        self.output_file = output_file
        self.header = header

    def get_data(self, j):
        d = []
        d.append(j['votes']['funny'])
        d.append(j['votes']['useful'])
        d.append(j['votes']['cool'])
        d.append(j['user_id'])
        d.append(j['review_id'])
        d.append(j['stars'])
        d.append(j['date'])
        d.append(j['text'].replace(',', ''))
        d.append(j['type'])
        d.append(j['business_id'])
        return d

    def transform_review_data(self):
        """
        This method is to consume the json format data of reviews from YELP and transform them into tabular format.
        :return:
            Save the transformed data to a text file.
        """
        review_content = read_text(self.path + '/' + self.input_file)
        with open(self.path + '/' + self.output_file, 'w') as f:
            csv_file = csv.writer(f, delimiter=',')
            csv_file.writerow(self.header)
            for item in review_content:
                js = convert_string_to_json(item)
                csv_file.writerow(self.get_data(js))

    def build_collection(self, doc, path, filename):
        tok = metapy.analyzers.ICUTokenizer(suppress_tags=True)
        tok = metapy.analyzers.LowercaseFilter(tok)
        tok = metapy.analyzers.ListFilter(tok, path + '/' + filename,
                                          metapy.analyzers.ListFilter.Type.Reject)
        tok = metapy.analyzers.Porter2Filter(tok)
        tok.set_content(doc.content())
        return tok

    def generate_collection_corpus(self):
        """
        Collect the words from dataset text column, count the frequency of each word (excluding stop words)
        Save to file
        :return: None
        """
        dataset = pandas.read_csv(self.path + '/' + self.output_file)
        X = dataset['text']
        # X = X[:50000]
        complete_set = ''
        for i in X:
            if isinstance(i, float):
                i = str(i)
            complete_set = complete_set + i

        doc = metapy.index.Document()
        doc.content(complete_set)
        tokens = self.build_collection(doc, self.dir_path, 'stopwords.txt')

        all_words = nltk.FreqDist(tokens)
        string = ''
        for word, frequency in all_words.most_common():
            string += u'{},{}\n'.format(word, frequency)

        write_to_file(self.output_path, 'word_frequency_reviews.txt', string)

        df = pandas.DataFrame(all_words.most_common())
        df.columns = ['Word', 'Freq']
        print(df)
        ax = df.plot(legend=True, title='Word frequency distribution (Zipf\'s law)')
        ax.set_xlabel('Words', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        plt.show()

    def star_to_attributes(self):
        ''' Read the dataset using pandas library
        Drop the row which does not have comment.
        describe the data by grouping per rating
        Classify the positive/negative based on ratings
        For Analysis: -> Positive as 3, 4, 5; Negative as 1, 2
        :return: None
        '''
        dataset = pandas.read_csv(self.path + '/' + self.output_file)
        dataset = dataset[dataset['text'].notnull()]
        print('Shape of the data set:', dataset.shape)
        print(dataset.describe())
        print(dataset.groupby('stars').size())
        show_plots(dataset, 'Purples_r', 'stars')

        # Analysis
        dataset['stars_attribute'] = dataset['stars'].apply(
            lambda x: 'Positive' if (3 <= int(x) <= 5) else 'Negative')
        # utils.save_dataset_to_csv(self.dataset, self.output_path, self.revised_file_sent)
        show_plots(dataset, 'copper_r', 'stars_attribute')


def main():
    st_time = datetime.datetime.now()
    input_file = 'yelp_academic_dataset_review.json'
    output_file = 'yelp_academic_dataset_review.csv'
    # Assumed every message(data per line) contains same key value pair.
    header = ['votes.funny', 'votes.useful', 'votes.cool', 'user_id', 'review_id', 'stars', 'date', 'text', 'type',
              'business_id']
    ra = YelpReviewsAnalysis(input_file, output_file, header)
    # ra.transform_review_data()    # uncomment this line to use the json to csv transformation for reviews file.
    # ra.generate_collection_corpus()       # Uncomment this line to generate the word frequency
    ra.star_to_attributes()
    en_time = datetime.datetime.now()
    print('Total execution time (milliseconds): ' + str((en_time - st_time).total_seconds() * 1000))


if __name__ == "__main__":
    main()
