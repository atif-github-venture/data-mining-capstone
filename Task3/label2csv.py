import pandas

import datetime

from utils import show_plots


def label2csv():
    st_time = datetime.datetime.now()
    column = ['name', 'label']
    dataset = pandas.read_csv('indian.label', delimiter="\t", names=column)
    dataset['name'].str.lower()
    print('Shape of the data set:', dataset.shape)
    print(dataset.describe())
    print(dataset.groupby('label').size())
    show_plots(dataset, 'Purples_r', 'label')
    dataset.to_csv(r'indian_label.csv')
    en_time = datetime.datetime.now()
    print('Total execution time (milliseconds): ' + str((en_time - st_time).total_seconds() * 1000))


def correct_label_analysis():
    st_time = datetime.datetime.now()
    column = ['name', 'original_label', 'ground_truth', 'correct_label', 'comment']
    dataset = pandas.read_csv('indian_label.csv', delimiter=",", names=column, header=0)
    print('Shape of the data set:', dataset.shape)
    print(dataset.describe())
    print(dataset.groupby('correct_label').size())
    show_plots(dataset, 'Wistia', 'correct_label')
    en_time = datetime.datetime.now()
    print('Total execution time (milliseconds): ' + str((en_time - st_time).total_seconds() * 1000))


# label2csv()
correct_label_analysis()
