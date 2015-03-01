def compare_sets(set1,set2):
    words_count=0
    sents_pointer=0
    words_pointer=0
    rate=0
    sents_tag_match_count_all=0
    sents_tag_match_count=0
    words_tag_match_count=0
    for sents_pointer in range(0,len(set1)):
        for words_pointer in range(0,len(set1[sents_pointer])):
             words_count=words_count+1
             if set1[sents_pointer][words_pointer][1]==set2[sents_pointer][words_pointer][1]:
                 words_tag_match_count=words_tag_match_count+1
                 words_tag_match_count_all=words_tag_match_count_all+1
        if words_tag_match_count==len(set1[sents_pointer]):
             sents_tag_match_count=sent_tag_match_count+1
             words_tag_match_count=0
    rate=words_tag_match_all/words_count             
    return rate
 