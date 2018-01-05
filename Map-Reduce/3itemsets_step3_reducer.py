#!/usr/bin/env python
import sys


current_items = None
current_count = 0
#itemset = None
item1 = None
item2 = None
item3 = None



#with open ('3itemsets_step3_output__nk','r') as out:
 #   for line in out:
for line in sys.stdin:
	# remove leading and trailing whitespace
        line = line.strip()
#        print(line)
        	# parse the input we got from mapper.py
        item1,item2,item3,count = line.split('\t',3)
    
        	# convert count (currently a string) to int
        try:
            count = int(count)
        except ValueError:
        		# count was not a number, so silently
        		# ignore/discard this line
          continue
                
        	# this IF-switch only works because Hadoop sorts map output
        	# by key (here: word) before it is passed to the reducer
        if current_items == (item1,item2,item3):            
            current_count += count
        else:
            if (current_items!=None) & (current_count >= 1000) :
        			# write result to STDOUT
                print('%s\t%s\t%s' % (current_items[0],current_items[1],current_items[2]))
                
            current_count = count
            current_items = (item1,item2,item3)
     
    ### do not forget to output the last word if needed!
if  (current_items == (item1,item2,item3)) & (current_count >= 1000):
    print('%s\t%s\t%s' % (current_items[0],current_items[1],current_items[2]))
