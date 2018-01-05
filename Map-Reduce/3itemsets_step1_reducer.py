#!/usr/bin/env python

import sys
import re
current_item = " "
item1=None
item2=None

first_item_list = []  
b=[]
c=[]
d=[]

i=0
h=0
k=0
i1=0
#with open('3itemsets_step1_output_map','r') as out:
for line in sys.stdin:
#    for line in out:   
	# remove leading and trailing whitespace
    line = line.strip()
            
        	# parse the input we got from mapper.py
    item1,item2 = line.split('\t', 1)       
            
    i1=item1+"|"+item2
    first_item_list.append(i1)
    len_first_item = len(first_item_list)
        
first_item_list.sort()
            
           
for i1 in range(0,len_first_item):
    for j1 in range(i+1,len_first_item):
        a=first_item_list[i1]
        a1=a.split("|")
        first_item_a = a1[0]
        second_item_a = a1[1]
        #           print(a1)
        b=first_item_list[j1]
        b1=b.split("|")
        first_item_b = b1[0]
        second_item_b = b1[1]

        if(first_item_a==first_item_b):
            print('%s\t%s\t%s' % (second_item_a,second_item_b,first_item_a))  
##========================================================================
#    i1=0
#    for i1 in range(0,len_first_item-2):
#        a=first_item_list[i1]
#        a1=a.split("|")
#        b=first_item_list[i1+1]
#        b1=b.split("|")
#        print("val",a,b)
#        if a1[0] == b1[0]:
#            print("in if")
#            d.append(a1[1])
#            d.append(b1[1])
#            current_item = item1
#        else:
#            print("in else")
#            i=0
#            l=len(d)
#                
#            for i in range(0,l):
#                for j in range(i+1,l):
#                    print(a1[0],d[i],d[j])
#            d.clear()
#            current_item = item1
            
#        for m in range(0,len(d)):
#            value = d[m]
#            third_item = d[m+1]
#            if current_item == item1:
#                print("hi")
##                print(d[m],current_item)
#            current_item = item1
#       
#    for i in range(len_items):
#        if current_item == item1:
#            print(current_item,d[i],d[i+1])  
##        else:
#            if current_item != a[i]:
#                print("hi")
#    current_item = item1

#                     
#       
#for k in first_item_list:
#    items1 = k.split["|"]
#    first_item = items1[0] +"|"+ items1[1]
#    for j in range(i+1,len_first_item):
#        items2 = a[j]
#        second_item = items2.split("|")
#        if first_item[0] == second_item[0]:
#            print(first_item[0],second_item[0],second_item[1])
#    
##    first_item = a[i]
##    cur_item,next_item=b[i],b[i+1]
##    i=0
#    for j in range(i+1,len(b)):
#        current_item == a[j]
#        if first_item[j] == current_item[j]:
##            for k in range(0,len(a)-1):
#            print(current_item[0],current_item[1],first_item[0])
#        current_item = item1
#        
##                k=a[i+1]
#        i+1
#        current_item = item1
#        print(h,k,item1)
#            a1=item1
#            a.append(a2)
#            b.append(a1)
##            b.append(item1)
##            print(item1,a1)
#            i+1
        
#        
#        for i in range(i+1,len(a)):
##            h=b[i]
##            k=b[i+1]
#            if a[0] == b[0]:
#                print(b[i],b[i+1],a[0])
#            i+1
#        i+1
#        current_item = item1
#        print(h,k,item1)


   
#if current_item == item1:
#    print(cur_item,next_item,item1)
    

#        if current_item == item1:
#        print(item1,item2,)

        

