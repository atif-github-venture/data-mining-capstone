import sys
import gensim
import logging

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class Sentences(object):
    stops = stopwords.words('english')

    def __init__(self, path):
        self.path = path

    def __iter__(self):
        with open(self.path) as fh:
            self.cnt = 0
            for line in fh:
                self.cnt += 1
                # if self.cnt > 10000: break
                line = line.strip()
                for sent in sent_tokenize(line):
                    yield [i.lower() for i in word_tokenize(sent) if i.lower() not in Sentences.stops]


def proceeseLabel(path):
    pos = []
    neg = []
    with open(path) as fh:
        for line in fh:
            token, label = line.strip().split(',')
            token = '_'.join(token.split())
            if int(label):
                pos.append(token)
            else:
                neg.append(token)
    return (pos, neg)


if __name__ == '__main__':

    type = 'analyse'
    inputPath = 'Indian_comments.txt'
    modelPath = 'model'
    labelPath = 'indian_label_tab.csv'

    if type == 'train':
        sentences = Sentences(inputPath)
        model = gensim.models.Word2Vec(sentences, workers=6)
        model.save(modelPath)

    elif type == 'train_phrase':
        ngramNum = 2
        sentences = Sentences(inputPath)
        ngram = None
        for i in range(ngramNum - 1):
            ngram = gensim.models.Phrases(sentences)

        model = gensim.models.Word2Vec(ngram[sentences], workers=6)
        model.save(modelPath)

    elif type == 'analyse':
        model = gensim.models.Word2Vec.load(modelPath)
        labelpos, labelneg = proceeseLabel(labelPath)
        labelpos = set(labelpos) & set(model.wv.vocab.keys())
        labelneg = set(labelneg) & set(model.wv.vocab.keys())

        mostSim = model.most_similar(
            positive=labelpos,
            negative=labelneg,
            topn=100000
        )

        newDishes = [i[0] for i in mostSim if len(i[0].split('_')) > 1]
        for dish in list(newDishes)[:20]:
            print(dish)
