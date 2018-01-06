# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
import math
import numpy as np
from nltk.tokenize import RegexpTokenizer    
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
#from operator import itemgetter

#Reference used
#https://docs.python.org/2/tutorial/datastructures.html - for dictionary and list
#https://google.github.io/styleguide/pyguide.html

# Following are the declarations
corpusroot = 'C:\presidential_debates'
tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
stemmer = PorterStemmer()

#List and set declaration
document_tokens = {}
idf = {}
doc_set = {}
document_frequency_1 = {}

all_docs_tokens_list = []

#Functions to stem tokens
def stem_tokens(tokens):
    stemmed= []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

#function to preprocess the string
def preprocess(qstring):
    qtokens = qstring.lower()
    qtokens = tokenizer.tokenize(qtokens)
    qtokens = [w for w in qtokens if not w in stopwords.words('english')]
    qtokens = stem_tokens(qtokens)

#Below to read files, tokenize 
for filename in os.listdir(corpusroot):
    file = open(os.path.join(corpusroot, filename),"r", encoding = 'UTF-8')    
    document = file.read()
    file.close()
    document = document.lower()
    terms = tokenizer.tokenize(document)  
    filter_all = [w for w in terms if not w in stopwords.words('english')]
    stem = stem_tokens(filter_all) 
    document_tokens[filename] = stem
    all_docs_tokens_list = all_docs_tokens_list + stem

#List of the tokens in all files
allwords_set =  set(all_docs_tokens_list)
allwords = list(allwords_set)
document_frequency = dict([x,0] for x in allwords)

#function for term count in the document
def term_count(doclist):
	allwords = list(set(doclist))
	tf_count = dict([x,0] for x in allwords)
	for i in tf_count:
		tf_count[i] = doclist.count(i)
	return tf_count

frequency = term_count(all_docs_tokens_list)
for doc in document_tokens:
	document_frequency_1[doc] = term_count(document_tokens[doc])

#Below determines the document frequency 
for words in allwords:
    for i in document_tokens:
        if words in document_tokens[i]:
            document_frequency[words] = document_frequency[words] + 1

#Below determines idf
for word in document_frequency:
    idf[word] = np.log10(30/document_frequency[word]) 

#below function gets the idf for token in the document: 
# Returns - 1 incase there is no token in the corpus
def getidf(token): 
    if token not in allwords:
        return -1
    else:
        return idf[token]    

#Below gets get the normalized tf.tdf values for all the terms in document 
normalized_tfidf = {}
document_length = {}
for doc in document_frequency_1:
    tf= {}
    tfidf = {}
    temp3 = {}
    temp = 0
    for term in document_frequency_1[doc]:
        if document_frequency_1[doc][term]:
            tf[term] = ((np.log10(document_frequency_1[doc][term]))+1)
            tfidf[term] = tf[term] * idf[term]
            temp = temp + tfidf[term]**2                         
        else:
            tfidf[term] = 0.0   
    doc_set[doc] = tfidf
    document_length[doc] = math.sqrt(temp)    
    for term in document_frequency_1[doc]:
        doc_set[doc] = tfidf
        temp3[term] = tfidf[term]/document_length[doc]
    
    normalized_tfidf[doc] = temp3
     
#Below function gets the weight token given in the filename
def getweight(filename,token):
    result = normalized_tfidf[filename].get(token)
    if result is None:
        result = 0.0
    return result

    
#https://docs.python.org/3/howto/sorting.html
#below gets the topten document for the given token
def topten(token,normalized_tfidf = normalized_tfidf):
    result_=[]
    posting_list = []
    for filename in normalized_tfidf:
        weight = normalized_tfidf[filename].get(token)
        if not weight:
            weight = 0.0
        temp = {'filename':filename,'weight':weight}
        result_.append(temp)
     
    posting_list = sorted(result_, key=lambda x: x['weight'], reverse=True)
#    print(posting_list[:10])
    return posting_list[:10]
       
#query function to retrieve top ten documents
def query(qstring): 
    temp = 0  
    q_vector_normalized = {}
    qtokens = qstring.lower()
    qtokens = tokenizer.tokenize(qtokens)
    qtokens = [w for w in qtokens if not w in stopwords.words('english')]
    qtokens = stem_tokens(qtokens)
    q_count= term_count(qtokens)
    q_vector = dict([x,np.log10(q_count[x])+1] for x in q_count)    

    for term in q_vector:
        temp = temp + q_vector[term]**2
        temp = math.sqrt(temp)    
        q_vector_normalized[term] = temp
    result_ =[]
    scores = {}
    seeninall = {}
    for token in q_vector:
        toplist = topten(token,10)
        if toplist is None: continue
        for filename in [item[0] for item in toplist]:
            if filename not in scores:
                scores[filename] = 0
                seeninall[filename] = True

        #temp = {token:topten(token)}
        #result_.append(temp)

 #   files = []

  #  for token in result_:
   #     print(token[0])
    #    files.append(set(token))

    #print(files)
    #print(set.intersection(*files))
    
#Check Output:
           
#print("(%s, %.12f)" % query("health insurance wall street"))

#print("(%s, %.12f)" % query("particular constitutional amendment"))

#print("(%s, %.12f)" % query("terror attack"))

#print("(%s, %.12f)" % query("vector entropy"))

print("%.12f" % getweight("2012-10-03.txt","health"))

print("%.12f" % getweight("1960-10-21.txt","reason"))

print("%.12f" % getweight("1976-10-22.txt","agenda"))

print("%.12f" % getweight("2012-10-16.txt","hispan"))

print("%.12f" % getweight("2012-10-16.txt","hispanic"))

print("%.12f" % getidf("health"))

print("%.12f" % getidf("agenda"))

print("%.12f" % getidf("vector"))

print("%.12f" % getidf("reason"))

print("%.12f" % getidf("hispan"))

print("%.12f" % getidf("hispanic"))

   