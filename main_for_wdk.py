{\rtf1\ansi\ansicpg1252\cocoartf1343\cocoasubrtf140
{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
\margl1440\margr1440\vieww10800\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural

\f0\fs24 \cf0 # CSE5525 NLP Homework 2 Group 1\
import nltk\
import numpy\
import math\
import itertools\
from nltk.corpus import treebank\
from viterbi import viterbi\
\
full_training_set = nltk.corpus.treebank.tagged_sents()[0:3500]\
training_set1 = full_training_set[0:1750]\
training_set2 = full_training_set[1750:]\
test_set = nltk.corpus.treebank.tagged_sents()[3500:]\
\
# Step 2: Retrieve P(W_i | T_i) and P(T_i| T_i-1)\
#\
# Sample Usage 1: \
#   print set1_cpd_tags['DT'].prob('JJ')\
#   meaning print the prob of adjective given determinor from training set 1\
#\
# Sample Usage 2: \
#   print full_cpd_word_tag['DT'].prob('the')\
#   meaning print the prob of word 'the' given determinor from full training set\
#\
# PS. cpd as the Conditional Prob Distribution\
# PSS. We use Laplace distribution for unseen cases\
\
# Full Traning Set\
full_training_set_words = []\
for sent in full_training_set:\
    full_training_set_words.append(('<s>','<s>'))\
    full_training_set_words.extend([ (tag, word) for (word, tag) in sent ])\
    full_training_set_words.append(('</s>','</s>'))\
\
full_tags = [tag for (tag, word) in full_training_set_words]\
full_words = [word for (tag, word) in full_training_set_words]\
\
full_cfd_word_tag = nltk.ConditionalFreqDist(full_training_set_words)\
full_cfd_tags = nltk.ConditionalFreqDist(nltk.bigrams(full_tags))\
    \
# Get all the tags and words\
for sent in test_set:\
    full_training_set_words.append(('<s>','<s>'))\
    full_training_set_words.extend([ (tag, word) for (word, tag) in sent ])\
    full_training_set_words.append(('</s>','</s>'))\
\
full_tags = [tag for (tag, word) in full_training_set_words]\
full_words = [word for (tag, word) in full_training_set_words]\
    \
full_tag_set = set(full_tags)\
full_word_set = set(full_words)\
\
num_tags = len(full_tag_set)\
num_words = len(full_word_set)\
\
dict_tags = \{\}\
dict_words = \{\}\
\
dict_tags['<s>'] = 0\
dict_words['<s>'] = 0\
temp = 1\
for tag in full_tag_set:\
    if tag != '<s>' and tag != '</s>':\
        dict_tags[tag] = temp\
        temp = temp + 1\
temp = 1\
for word in full_word_set:\
    if word != '<s>' and word != '</s>':\
        dict_words[word] = temp\
        temp = temp + 1 \
dict_tags['</s>'] = num_tags - 1\
dict_words['</s>'] = num_words - 1\
\
# Traning Set 1\
set1_training_set_words = []\
for sent in training_set1:\
    set1_training_set_words.append(('<s>','<s>'))\
    set1_training_set_words.extend([ (tag, word) for (word, tag) in sent ])\
    set1_training_set_words.append(('</s>','</s>'))\
    \
set1_tags = [tag for (tag, word) in set1_training_set_words]\
set1_words = [word for (tag, word) in set1_training_set_words]\
\
set1_cfd_word_tag = nltk.ConditionalFreqDist(set1_training_set_words)\
set1_cfd_tags = nltk.ConditionalFreqDist(nltk.bigrams(set1_tags))\
\
# eliminate wrong counts\
full_cfd_tags['</s>']['<s>'] = 0\
set1_cfd_tags['</s>']['<s>'] = 0\
\
# Get the transition prob\
A_full_table = numpy.zeros((num_tags, num_tags), dtype = 'double')\
A_set1_table = numpy.zeros((num_tags, num_tags), dtype = 'double')\
\
for tag_1 in full_tag_set:\
    if tag_1 != '</s>':\
        full_num = 0\
        set1_num = 0\
        for tag_2 in full_tag_set:\
            if tag_2 != '<s>' and (tag_1 != '<s>' or tag_2 != '</s>'):\
                if full_cfd_tags[tag_1][tag_2] == 0:\
                    full_cfd_tags[tag_1][tag_2] = 1\
                if set1_cfd_tags[tag_1][tag_2] == 0:\
                    set1_cfd_tags[tag_1][tag_2] = 1\
                full_num = full_num + full_cfd_tags[tag_1][tag_2]\
                set1_num = set1_num + set1_cfd_tags[tag_1][tag_2]\
        for tag_2 in full_tag_set:\
            A_full_table[dict_tags[tag_1]][dict_tags[tag_2]] = float(full_cfd_tags[tag_1][tag_2])/float(full_num)\
            A_set1_table[dict_tags[tag_1]][dict_tags[tag_2]] = float(set1_cfd_tags[tag_1][tag_2])/float(set1_num)\
            \
# Get the emission probs\
B_full_table = numpy.zeros((num_tags, num_words), dtype = 'double')\
B_set1_table = numpy.zeros((num_tags, num_words), dtype = 'double')\
\
B_full_table[dict_tags['<s>']][dict_words['<s>']] = 1.0\
B_full_table[dict_tags['</s>']][dict_words['</s>']] = 1.0\
B_set1_table[dict_tags['<s>']][dict_words['<s>']] = 1.0\
B_set1_table[dict_tags['</s>']][dict_words['</s>']] = 1.0\
\
for tag in full_tag_set:\
    if tag != '</s>' and tag != '<s>':\
        full_num = 0\
        set1_num = 0\
        for word in full_word_set:\
            if word != '</s>' and word != '<s>':\
                if full_cfd_word_tag[tag][word] == 0:\
                    full_cfd_word_tag[tag][word] = 1\
                if set1_cfd_word_tag[tag][word] == 0:\
                    set1_cfd_word_tag[tag][word] = 1\
                full_num = full_num + full_cfd_word_tag[tag][word]\
                set1_num = set1_num + set1_cfd_word_tag[tag][word]\
        for word in full_word_set:\
            B_full_table[dict_tags[tag]][dict_words[word]] = float(full_cfd_word_tag[tag][word])/float(full_num)\
            B_set1_table[dict_tags[tag]][dict_words[word]] = float(set1_cfd_word_tag[tag][word])/float(set1_num)\
            \
\
#step3 viterbi\
\
# The init_table is the initial probabilities and enforces every sentence to start with <s>\
init_table = \{\}\
for tag in full_tag_set:\
    init_table[tag] = 0.00000000000000000001\
init_table['<s>'] = 1.0\
\
# test_obs is a sentence to test viterbi\
test_obs = ['<s>', 'Pierre', 'Vinken', ',' , '61' , "years", "old", "will", "join", "the", "board", "as", "a", "nonexecutive", "director", "Nov.", "29","."]\
\
#run viterbi\
print viterbi(dict_tags, dict_words, test_obs, full_tag_set, init_table, A_full_table, B_full_table )\
def tagging(set_,tags,word_tag):\
    global full_tags\
    global init_table\
    global full_cpd_tags\
    global full_cpd_word_tag\
    tagset=[]\
    for i in range (0,len(set_)):\
        tagset.append(dict_tags, dict_words, viterbi(set_[i], set(full_tags), init_table, tags, word_tag))\
    return combine_tag_word(set_,tagset)\
########\
def compare_sets(set1,set2,indicator): \
    """when indicator==words_match_rate,this function will return the rate that words are tagged correctly\
when indicator== sents_match_rate, this function will return the rate sentences are tagged correctly"""\
    words_count=0 \
    # the number of words being compared \
    sents_pointer=0 \
    # which sentence is being compared\
    words_pointer=0\
    #  which word is being compared\
    words_match_rate=0\
     #the rate words are tagged correctly\
    sents_match_rate=0\
     #the rate sentences are tagged correctly\
    words_tag_match_count_all=0\
     # the number of words that are tagged correctly\
    sents_tag_match_count=0\
     # the number of sentences that are tagged correctly\
    words_tag_match_count=0\
     # the number of words that are tagged correctly in the sentence being compared\
    sents_count=0\
    words_count_eachsentence=0\
    if len(set1)<len(set2):\
        sents_count=len(set1)\
    else:\
        sents_count=len(set2)\
    for sents_pointer in range(0,sents_count):\
        if len(set1[sents_pointer])<len(set2[sents_pointer]):\
             words_count_eachsentence=len(set1[sents_pointer])\
        else:\
            words_count_eachsentence=len(set2[sents_pointer])\
        for words_pointer in range(0,words_count_eachsentence):\
             words_count=words_count+1\
             if set1[sents_pointer][words_pointer][1]==set2[sents_pointer][words_pointer][1]and set1[sents_pointer][words_pointer][0]==set2[sents_pointer][words_pointer][0]:\
                 words_tag_match_count=words_tag_match_count+1\
                 words_tag_match_count_all=words_tag_match_count_all+1\
        if words_tag_match_count==len(set1[sents_pointer]):\
             sents_tag_match_count=sents_tag_match_count+1\
             words_tag_match_count=0\
    words_match_rate=words_tag_match_count_all/float(words_count)\
    sents_match_rate=sents_tag_match_count/float(sents_count)\
    if indicator=="words_match_rate":            \
      return words_match_rate\
    elif indicator=="sents_match_rate":\
      return sents_match_rate\
 #######}