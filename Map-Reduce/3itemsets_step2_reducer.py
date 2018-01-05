#!/usr/bin/env python
import sys

current_item = None
current_item2 = None


itemsets_2_list=[]
itemsets_3_list=[]
i=0

#with open ('3itemsets_step2_output_map123') as out:
#    for line in out:
for line in sys.stdin:
	# remove leading and trailing whitespace
        line = line.strip()
    
    	# parse the input we got from mapper.py
    
        item1,item2, item3 = line.split('\t', 2)
        
        if item3 == "U":
            a=item1+"|"+item2
            itemsets_2_list.append(a)
            count_b = len(itemsets_2_list)
        else:            
            c= item1+"|"+item2+"|"+item3
            itemsets_3_list.append(c)
            count_c = len(itemsets_3_list)
            
for j in itemsets_3_list:
    j_split=j.split("|")
    j2=j_split[0]+"|"+ j_split[1]
    
    if j2 in itemsets_2_list:        
        print('%s\t%s\t%s'%(j_split[0],j_split[1],j_split[2]))        
