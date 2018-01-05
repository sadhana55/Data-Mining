Dataset
We use a corpus of all the general election presidential debates from 1960 to 2012. We processed the corpus and provided you a .zip file, 
which includes 30 .txt files. Each of the 30 files contains the transcript of a debate and is named by the date of the debate
TASK
 (1) Read the 30 .txt files, each of which has the transcript of a presidential debate. 
(2) Tokenize the content of each file.]
(3) Perform stopword removal on the obtained tokens.
NLTK already comes with a stopword list, as a corpus in the "NLTK Data" (http://www.nltk.org/nltk_data/). 
You need to install this corpus. Follow the instructions at http://www.nltk.org/data.html.
You can also find the instruction in this book:http://www.nltk.org/book/ch01.html (Section 1.2 Getting Started with NLTK). 
Basically, use the following statements in Python interpreter. A pop-up window will appear. 
Click "Corpora" and choose "stopwords" from the list.
(4) Also perform stemming on the obtained tokens. NLTK comes with a Porter stemmer. 
Try the following code and learn how to use the stemmer.

from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
print(stemmer.stem('studying'))
(5) Using the tokens, compute the TF-IDF vector for each document. Use the following equation that we learned in the lectures 
to calculate the term weights, in which tt is a token and dd is a document:
wt,d=(1+log10tft,d)×(log10Ndft).wt,d=(1+log10tft,d)×(log10Ndft).
Note that the TF-IDF vectors should be normalized (i.e., their lengths should be 1).
Represent a TF-IDF vector by a dictionary. The following is a sample TF-IDF vector.
{'sanction': 0.014972337775895645, 'lack': 0.008576372825970286, 'regret': 0.009491784747267843, 'winter': 0.030424375278541155}

(6) Given a query string, calculate the query vector. (Remember to convert it to lower case.) 
In calculating the query vector, don't consider IDF. I.e., use the following equation to calculate the term weights in the query vector, 
in which tt is a token and qq is the query:

wt,q=(1+log10tft,q).wt,q=(1+log10tft,q).
The vector should also be normalized.

