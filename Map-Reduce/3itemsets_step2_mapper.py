#!/usr/bin/env python

import sys
import re


#with open ('2itemsets_output') as out:
#    for line in out:   
for line in sys.stdin:          
            # remove leading and trailing whitespace
        line = line.strip()
        items = re.findall(r"[\'0-9]+", line)
        len_items=len(items)
        if len_items == 2:
            print('%s\t%s\t%s' %(items[0],items[1],"U"))
        else:
            print('%s\t%s\t%s' %(items[0],items[1],items[2]))
        