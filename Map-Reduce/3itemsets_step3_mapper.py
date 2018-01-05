#!/usr/bin/env python
##
import sys
import re
##
#with open ('retail.dat') as out:
 #   for line1 in out:
for line1 in sys.stdin:
        	# remove leading and trailing whitespace
        line1 = line1.strip()
        
        with open ('candidates') as cand:
            for line_c in cand:
                        
                line_c = line_c.strip()
                        
                item1,item2,item3 = line_c.split()
        #        print(item1,item2,item3)
                
                if item1 in line1 and item2 in line1 and item3 in line1:
                    print('%s\t%s\t%s\t%s' %(item1,item2,item3,1))
                    
                    
            