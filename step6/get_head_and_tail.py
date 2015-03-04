def get_head_and_tail(set_):
	set_f=[]
        set_f_all=[]
	for sent in set_:
		set_f.append(('<s>','<s>'))
		set_f.extend([ (word, tag) for (word, tag) in sent ])
		set_f.append(('</s>','</s>'))
                set_f_all.append(set_f)
                set_f=[]
	return set_f_all
