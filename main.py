# CSE5525 NLP Homework 2 Group 1
import nltk
import itertools
from nltk.corpus import treebank

full_training_set = nltk.corpus.treebank.tagged_sents()[0:3500]
training_set1 = full_training_set[0:1750]
training_set2 = full_training_set[1750:]
test_set = nltk.corpus.treebank.tagged_sents()[3500:]

# Step 2: Retrieve P(W_i | T_i) and P(T_i| T_i-1)
#
# Sample Usage 1: 
#	print set1_cpd_tags['DT'].prob('JJ')
# 	meaning print the prob of adjective given determinor from training set 1
#
# Sample Usage 2: 
#	print full_cpd_word_tag['DT'].prob('the')
# 	meaning print the prob of word 'the' given determinor from full training set
#
# PS. cpd as the Conditional Prob Distribution
# PSS. We use Laplace distribution for unseen cases

# Full Traning Set
full_training_set_words = []
for sent in full_training_set:
	full_training_set_words.append(('S','S'))
	full_training_set_words.extend([ (tag, word) for (word, tag) in sent ])
	full_training_set_words.append(('T','T'))

full_cfd_word_tag = nltk.ConditionalFreqDist(full_training_set_words)
full_cpd_word_tag = nltk.ConditionalProbDist(full_cfd_word_tag, nltk.LaplaceProbDist)

full_tags = [tag for (tag, word) in full_training_set_words]
full_cfd_tags = nltk.ConditionalFreqDist(nltk.bigrams(full_tags))
full_cpd_tags = nltk.ConditionalProbDist(full_cfd_tags, nltk.LaplaceProbDist)

#Traning Set 1
set1_training_set_words = []
for sent in training_set1:
	set1_training_set_words.append(('S','S'))
	set1_training_set_words.extend([ (tag, word) for (word, tag) in sent ])
	set1_training_set_words.append(('T','T'))

set1_cfd_word_tag = nltk.ConditionalFreqDist(set1_training_set_words)
set1_cpd_word_tag = nltk.ConditionalProbDist(set1_cfd_word_tag, nltk.LaplaceProbDist)

set1_tags = [tag for (tag, word) in set1_training_set_words]
set1_cfd_tags = nltk.ConditionalFreqDist(nltk.bigrams(set1_tags))
set1_cpd_tags = nltk.ConditionalProbDist(set1_cfd_tags, nltk.LaplaceProbDist)

print set1_cpd_tags['NN'].prob("T")

# Step 3

# Step 4

# Step 5

# Step 6
