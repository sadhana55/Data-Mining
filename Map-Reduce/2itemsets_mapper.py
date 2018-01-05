#!/usr/bin/env python

import sys
import re

for line in sys.stdin:
        	# remove leading and trailing whitespace
        line = line.strip()
            	# split the line into words
        items = re.findall(r"[\'0-9]+", line)
            	# increase counters
        len_items = len(items)
#        print(len_items)
        for i in range(len_items):
            for j in range(i+1,len_items):
            		# write the results to STDOUT (standard output);
            		# what we output here will be the input for the
            		# Reduce step, i.e. the input for reducer.py
            		#
            		# tab-delimited; the trivial word count is 1
              print('%s\t%s\t%s' % (items[i],items[j], 1))
