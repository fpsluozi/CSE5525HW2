def combine_tag_word(set_tag,set_word):
	lenth=0
	combined_array= []
	if len(set_word)<len(set_tag):
		lenth=len(set_word)
	else:
		lenth=len(set_tag)
	for i in range(0,lenth):
			combined_array.append((set_word[i],set_tag[i]))
	return combined_array