# CSE5525 NLP Homework 2 Group 1
import nltk
from nltk.corpus import treebank

full_training_set = nltk.corpus.treebank.tagged_sentences()[0:3500]
training_set1 = full_training_set[0:1750]
training_set2 = full_training_set[1750:]
test_set = nltk.corpus.treebank.tagged_sentences()[3500:]
