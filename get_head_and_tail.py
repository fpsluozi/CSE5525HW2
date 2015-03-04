obs = [((u'Carnival', u'NNP'), (u'said', u'VBD'), (u'0', u'-NONE-'), (u'the', u'DT'), (u'Fantasy', u'NNP'), (u',', u','), (u'a', u'DT'), (u'2,050-passenger', u'JJ'), (u'ship', u'NN'), (u'that', u'WDT'), (u'*T*-3', u'-NONE-'), (u'was', u'VBD'), (u'slated', u'VBN'), (u'*-4', u'-NONE-'), (u'to', u'TO'), (u'be', u'VB'), (u'delivered', u'VBN'), (u'*-2', u'-NONE-'), (u'this', u'DT'), (u'month', u'NN'), (u',', u','), (u'will', u'MD'), (u'be', u'VB'), (u'delivered', u'VBN'), (u'*-1', u'-NONE-'), (u'in', u'IN'), (u'January', u'NNP'), (u'.', u'.')),(('q','w'),('e','r'))]

def get_head_and_tail(set_):
	set_f=[]
	for sent in full_training_set:
		set_f.append(('<s>','<s>'))
	    set_f.extend([ (tag, word) for (word, tag) in sent ])
        set_f.append(('</s>','</s>'))
	return set_f