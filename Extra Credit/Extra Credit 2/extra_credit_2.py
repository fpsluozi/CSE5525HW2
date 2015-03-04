# CSE5525 NLP Homework 2 Group 1
import nltk
import itertools
from nltk.corpus import treebank

#
# generate the file using the sentences in full_training_set
#

full_training_set = nltk.corpus.treebank.tagged_sents()[0:3500]

# output the sentences into file 'data'

file_object = open('data', 'w')

for sent in full_training_set:
    s = ''
    for (word, tag) in sent:
        s = s + tag + ' '
    file_object.writelines(s+'\n')