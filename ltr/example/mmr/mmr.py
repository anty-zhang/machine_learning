import os
import re
import time
from termcolor import colored
import sys
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import operator

# create stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()


def load_stopwords():
    stop_word_list = []
    with open('stopword.txt', 'r') as f:
        for word in f.readlines():
            stop_word_list.append(word.strip())
    return stop_word_list


stopwords = load_stopwords()

# print(stopwords)


def cleanData(sentence):
    # sentence = re.sub('[^A-Za-z0-9 ]+', '', sentence)
    # sentence filter(None, re.split("[.!?", setence))
    ret = []
    # 获取每个句子的摘要
    sentence = stemmer.stem(sentence)
    for word in sentence.split():
        if word not in stopwords:
            ret.append(word)
    return " ".join(ret)


def getVectorSpace(cleanSet):
    vocab = {}
    for data in cleanSet:
        for word in data.split():
            vocab[data] = 0
    return vocab.keys()


def calculateSimilarity(sentence, doc):
    if doc == []:
        return 0
    vocab = {}
    for word in sentence.split():
        vocab[word] = 0

    docInOneSentence = ''
    for t in doc:
        docInOneSentence += (t + ' ')
        for word in t.split():
            vocab[word] = 0

    cv = CountVectorizer(vocabulary=vocab.keys())

    docVector = cv.fit_transform([docInOneSentence])
    sentenceVector = cv.fit_transform([sentence])
    return cosine_similarity(docVector, sentenceVector)[0][0]


data = open('news_data.txt', 'r')
texts = data.readlines()

sentences = []
clean = []
originalSentenceOf = {}


start = time.time()

# Data cleansing
for line in texts:
    parts = line.strip().split('.')
    for part in parts:
        if part:
            cl = cleanData(part)
            sentences.append(part)
            clean.append(cl)
            originalSentenceOf[cl] = part

print("sentences=", sentences)
print("clean=", clean)
print("originalSentenceOf=", originalSentenceOf)
setClean = set(clean)

# calculate Similarity score each sentence with whole documents
scores = {}
for data in clean:
    temp_doc = setClean - set([data])
    score = calculateSimilarity(data, list(temp_doc))
    scores[data] = score
print(scores)


# calculate MMR
n = 20 * len(sentences) / 100
alpha = 0.5
summarySet = []
while n > 0:
    mmr = {}
    # kurangkan dengan set summary
    for sentence in scores.keys():
        if sentence not in summarySet:
            mmr[sentence] = alpha * scores[sentence] - (1 - alpha) * calculateSimilarity(sentence, summarySet)
    # selected = max(mmr.iteritems(), key=operator.itemgetter(1))[0]
    selected = max(mmr.items(), key=operator.itemgetter(1))[0]
    summarySet.append(selected)
    n -= 1


print('\nSummary:\n')
for sentence in summarySet:
    print(originalSentenceOf[sentence].lstrip(' '))

print('=============================================================')
print('\nOriginal Passages:\n')


for sentence in clean:
    if sentence in summarySet:
        print(colored(originalSentenceOf[sentence].lstrip(' '), 'red'))
    else:
        print(originalSentenceOf[sentence].lstrip(' '))
