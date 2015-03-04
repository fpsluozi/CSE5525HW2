def combine_tag_word(set_word,set_tag):
	lenth=0
        sents_lenth=0
	combined_array= []
        combined_array_all=[]
	if len(set_word)<len(set_tag):
		lenth=len(set_word)
	else:
		lenth=len(set_tag)
	for i in range(0,lenth):
            if len(set_word[i])<len(set_tag[i]):
                sents_lenth=len(set_word[i])
            else:
                sents_lenth=len(set_tag[i])
            for j in range(0,sents_lenth):
			combined_array.append((set_word[i][j],set_tag[i][j]))
            combined_array_all.append(combined_array)
            combined_array=[]
	return combined_array_all
