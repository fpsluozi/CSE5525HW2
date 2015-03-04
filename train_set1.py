# Traning Set 
def train_set1(training_set1):
    set1_training_set_words = []
    for sent in training_set1:
        set1_training_set_words.append(('<s>','<s>'))
        set1_training_set_words.extend([ (tag, word) for (word, tag) in sent ])
        set1_training_set_words.append(('</s>','</s>'))
    set1_tags = [tag for (tag, word) in set1_training_set_words]
    set1_words = [word for (tag, word) in set1_training_set_words]
    set1_cfd_word_tag = nltk.ConditionalFreqDist(set1_training_set_words)
    set1_cfd_tags = nltk.ConditionalFreqDist(nltk.bigrams(set1_tags))      
    set1_cpd_word_tag = nltk.ConditionalProbDist(set1_cfd_word_tag, nltk.MLEProbDist)
    set1_cpd_tags = nltk.ConditionalProbDist(set1_cfd_tags, nltk.MLEProbDist)