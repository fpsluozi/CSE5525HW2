# CSE5525 NLP Homework 2 Group 1
import nltk
from nltk.corpus import treebank

full_training_set = nltk.corpus.treebank.tagged_sents()[0:3500]
training_set1 = full_training_set[0:1750]
training_set2 = full_training_set[1750:]
test_set = nltk.corpus.treebank.tagged_sents()[3500:]

print full_training_set

dict_words = {}
dict_tags = {}
num_words = 0
num_tags = 0

for sent in nltk.corpus.treebank.tagged_sents():
    for word_tag in sent:
        word = word_tag[0]
        tag = word_tag[1]
        if dict_words.get(word) == None:
            dict_words[word] = num_words
            num_words = num_words + 1
        if dict_tags.get(tag) == None:
            dict_tags[tag] = num_tags
            num_tags = num_tags + 1

import numpy
t_w_table = numpy.zeros((num_tags, num_words))
t_t_table = numpy.zeros((num_tags, num_tags))