def train_fullset():
    full_training_set_words = []
    for sent in full_training_set:
        full_training_set_words.append(('<s>','<s>'))
        full_training_set_words.extend([ (tag, word) for (word, tag) in sent ])
        full_training_set_words.append(('</s>','</s>'))
    full_tags = [tag for (tag, word) in full_training_set_words]
    full_words = [word for (tag, word) in full_training_set_words]
    full_tag_set = set(full_tags)
    full_cfd_word_tag = nltk.ConditionalFreqDist(full_training_set_words)
    full_cfd_tags = nltk.ConditionalFreqDist(nltk.bigrams(full_tags))
    full_obs_set = []
    for sent in full_training_set:
        full_obs_set.append([ word for (word, tag) in sent ])
    for i in xrange(len(full_obs_set)):
        full_obs_set[i].append('</s>')
        full_obs_set[i].insert(0, '<s>')       
    full_cpd_word_tag = nltk.ConditionalProbDist(full_cfd_word_tag, nltk.MLEProbDist)
    full_cpd_tags = nltk.ConditionalProbDist(full_cfd_tags, nltk.MLEProbDist)
