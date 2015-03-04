# CSE5525 NLP Homework 2 Group 1
import nltk
import numpy
import itertools
from get_test_word import get_test_word
from viterbi import viterbi
from nltk.corpus import treebank
from get_test_word_firsttime import get_test_word_firsttime
from compare_function import compare_sets
from combine_tag_word import combine_tag_word
from get_head_and_tail import get_head_and_tail

full_training_set = nltk.corpus.treebank.tagged_sents()[0:3500]
training_set1 = full_training_set[0:1750]
training_set2 = full_training_set[1750:]
test_set = nltk.corpus.treebank.tagged_sents()[3500:]

# Step 2: Retrieve P(W_i | T_i) and P(T_i| T_i-1)
#
# Sample Usage 1: 
#   print set1_cpd_tags['DT'].prob('JJ')
#   meaning print the prob of adjective given determinor from training set 1
#
# Sample Usage 2: 
#   print full_cpd_word_tag['DT'].prob('the')
#   meaning print the prob of word 'the' given determinor from full training set
#
# PS. cpd as the Conditional Prob Distribution
# PSS. We use Laplace distribution for unseen cases

# Full Traning Set
full_tags = []
full_words =[]
full_tag_set=[]
full_cpd_word_tag =[]
full_cpd_tags = []
full_cfd_word_tag=[] 
full_cfd_tags =[]
set1_tags = []
set1_words = [] 
set1_cfd_word_tag =[]
set1_cfd_tags = []
set1_cpd_word_tag =[]
set1_cpd_tags =[]

#train full set
def train_fullset(x):
    global full_tags
    global full_words
    global full_tag_set
    global full_cpd_word_tag
    global full_cpd_tags
    global full_cfd_word_tag 
    global full_cfd_tags
    global full_training_set
    full_training_set_words = []
    for sent in full_training_set:
        if x==0:
            full_training_set_words.append(('<s>','<s>'))
        full_training_set_words.extend([ (tag, word) for (word, tag) in sent ])
        if x==0:
            full_training_set_words.append(('</s>','</s>'))
    full_tags = [tag for (tag, word) in full_training_set_words]
    full_words = [word for (tag, word) in full_training_set_words]
    full_tag_set = set(full_tags)
    full_cfd_word_tag = nltk.ConditionalFreqDist(full_training_set_words)
    full_cfd_tags = nltk.ConditionalFreqDist(nltk.bigrams(full_tags))
    full_obs_set = []
    for sent in full_training_set:
        full_obs_set.append([ word for (word, tag) in sent ])
    if x==0:
        for i in xrange(len(full_obs_set)):
            full_obs_set[i].append('</s>')
            full_obs_set[i].insert(0, '<s>')       
    full_cpd_word_tag = nltk.ConditionalProbDist(full_cfd_word_tag, nltk.MLEProbDist)
    full_cpd_tags = nltk.ConditionalProbDist(full_cfd_tags, nltk.MLEProbDist)

# Traning Set1 
def train_set1(x):
    global training_set1
    global set1_tags
    global set1_words 
    global set1_cfd_word_tag
    global set1_cfd_tags
    global set1_cpd_word_tag
    global set1_cpd_tags
    set1_training_set_words = []
    for sent in training_set1:
        if x==0:
            set1_training_set_words.append(('<s>','<s>'))
        set1_training_set_words.extend([ (tag, word) for (word, tag) in sent ])
        if x==0:
            set1_training_set_words.append(('</s>','</s>'))
    set1_tags = [tag for (tag, word) in set1_training_set_words]
    set1_words = [word for (tag, word) in set1_training_set_words]
    set1_cfd_word_tag = nltk.ConditionalFreqDist(set1_training_set_words)
    set1_cfd_tags = nltk.ConditionalFreqDist(nltk.bigrams(set1_tags))      
    set1_cpd_word_tag = nltk.ConditionalProbDist(set1_cfd_word_tag, nltk.MLEProbDist)
    set1_cpd_tags = nltk.ConditionalProbDist(set1_cfd_tags, nltk.MLEProbDist)
# train fullset and set1
train_fullset(0)
train_set1(0)

# Step 3 Viterbi
# Since both emission prob and transition prob can be zero, we use an extremely small epsilon
# to rule out zero probs during log calculation.

# The init_table is the initial probabilities and enforces every sentence to start with <s>
init_table = {}
for tag in full_tag_set:
    init_table[tag] = 0.00000000000000000001
init_table['<s>'] = 1.0

# test_obs is a sentence to test viterbi

# get test word
test_obs=get_test_word_firsttime(test_set)

# Step 4
def tagging(set_,tags,word_tag):
    global full_tags
    global init_table
    global full_cpd_tags
    global full_cpd_word_tag
    tagset=[]
    for i in range (0,len(set_)):
        tagset.append(viterbi(set_[i], set(full_tags), init_table, tags, word_tag))
    return combine_tag_word(set_,tagset)

# Step 5
def converge_test():
	#train
        global training_set1
        global training_set2
        global full_training_set
	train_set1(0)
	temp=tagging(get_test_word_firsttime(training_set2),set1_cpd_tags, set1_cpd_word_tag)
	print compare_sets(temp,get_head_and_tail(training_set2),"words_match_rate")
	for i in range(0,10):
            training_set2=temp
	    full_tarining_set=get_head_and_tail(training_set1)+training_set2
	    train_fullset(1)
            temp=tagging(get_test_word(training_set2),full_cpd_tags, full_cpd_word_tag)
	    print compare_sets(temp,training_set2,"words_match_rate")

converge_test()
