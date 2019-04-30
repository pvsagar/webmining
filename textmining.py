
# coding: utf-8

# # Assignment 4

# ### Q1: Define a tokenize function which does the following in sequence:
# - takes a string as an input
# - converts the string into lowercase
# - segments the lowercased string into tokens.
# #### A token is defined as follows:
# - Each token has at least two characters.
# - The first/last character can only be a letter (i.e. a-z) or a number (0-9)
# - In the middle, there are 0 or more characters, which can only be letters (a-z),numbers (0-9), hyphens ("-"), underscores ("_"), dot ("."), or "@" symbols.
# - lemmatizes all tokens using WordNetLemmatizer
# - removes stop words from the tokens (use English stop words list from NLTK)
# - generate token frequency dictionary, where each unique token is a key and the frequency of the token is the value. (Hint: you can use nltk.FreqDist to create it)
# - returns the token frequency dictionary as the output
# - Note, this question is similar to Q1 in your Assignment 1, but more complicated

# ### Q2: Find duplicate questions by similarity
# A data file 'qa.csv' has been provided for this question. This dataset has two columns: question and answer
# as shown in screenshot blow. Here we only use "question" column.
# - Define a function find_similar_doc as follows:
#     -takes two inputs: a list of documents as strings (i.e. docs), and the index of a selected document as an integer (i.e. doc_id).
#     - uses the "tokenize" function defined in Q1 to tokenize each document
#     - generates tf_idf matrix from the tokens (hint: reference to the tf_idf function defined in Section 7.5 in lecture notes)
#     - calculates the pairwise cosine distance of documents using the tf_idf matrix
#     - for the selected document, finds the index of the most similar document (but not the selected document itself!) by the cosine similarity score
#     - returns the index of the most similar document and the similarity score
#     
# - Test your function with two selected questions 15 and 51 respectively, i.e., doc_id = 15 and doc_id = 51.
#     - Check the most similar questions discovered for each of them
#     - Do you think this function can successfully find duplicate questions? Why does it work or not work? Write down your analysis in a document and upload it to canvas along with your code.

# In[76]:


import nltk
import re 
import string
from nltk.corpus import stopwords
import pandas as pd
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import wordnet
from sklearn.preprocessing import normalize
import numpy as np 
from scipy.spatial import distance

wordnet_lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(pos_tag):
    
    # if pos tag starts with 'J'
    if pos_tag.startswith('J'):
        # return wordnet tag "ADJ"
        return wordnet.ADJ
    
    # if pos tag starts with 'V'
    elif pos_tag.startswith('V'):
        # return wordnet tag "VERB"
        return wordnet.VERB
    
    # if pos tag starts with 'N'
    elif pos_tag.startswith('N'):
        # return wordnet tag "NOUN"
        return wordnet.NOUN
    
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        # be default, return wordnet tag "NOUN"
        return wordnet.NOUN

    
# Q1
def tokenize(text):
    text = text.lower()
    pattern=r'\w[\w\'_.@-]*\w'
    tokens = nltk.regexp_tokenize(text, pattern)
    tokens=[token.strip(string.punctuation) for token in tokens]
    tokens=[token.strip() for token in tokens if token.strip()!='']
    #print(len(tokens))                   
    #print(tokens)
    tagged_tokens= nltk.pos_tag(tokens)
    stop_words = stopwords.words('english')
    stop_words+=["they'll", "can't"]    
    lemmatized_words=[wordnet_lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for (word, tag) in tagged_tokens if word not in stop_words and word not in string.punctuation]
    token_count=nltk.FreqDist(lemmatized_words)
    return token_count

# Q2

def find_similar_doc(doc_id, docs):
    
#     print(doc_id)
#     print(docs)
    
    docs_tokens={idx:tokenize(doc) for idx,doc in enumerate(docs)}
#     print(docs_tokens)
    
    dtm=pd.DataFrame.from_dict(docs_tokens, orient="index" )
    dtm=dtm.fillna(0)
    
    tf=dtm.values
    doc_len=tf.sum(axis=1)
    tf=np.divide(tf.T, doc_len).T
    
    df=np.where(tf>0,1,0)
    
    idf=np.log(np.divide(len(docs),np.sum(df, axis=0)))+1
    smoothed_idf=np.log(np.divide(len(docs)+1, np.sum(df, axis=0)+1))+1    
    smoothed_tf_idf=normalize(tf*smoothed_idf)
    
    tf_idf=normalize(tf*idf)
    
    similarity = 1-distance.squareform(distance.pdist(tf_idf, 'cosine'))
#     print(similarity)
   
    best_matching_doc_id = np.argsort(similarity)[:,::-1][int(doc_id),1:2]
#     print(best_matching_doc_id)
    
    return best_matching_doc_id, similarity


if __name__ == "__main__":
# Test Q1
    text='''contact Yahoo! at "http://login.yahoo.com", select forgot
your password. If that fails to reset, contact Yahoo! at
their password department 408-349-1572 -- Can't promise
their phone department will fix, but they'll know where to
go next. Corporate emails from Yahoo! don't come from
their free mail system address space. Webmaster@yahoo.com
is not a corporate email address.'''
    print("Test Q1")
    for key, value in tokenize(text).items():
        print(key, value)
# You should get the result look like :
# contact 2 yahoo 3 http 1 login.yahoo.com 1
# select 1 forget 1 password 2 fail 1
# reset 1 department 2 408-349-1572 1 promise 1
# phone 1 fix 1 know 1 go 1
# next 1 corporate 2 email 2 come 1
# free 1 mail 1 system 1 address 2
# space 1 webmaster@yahoo.com 1
    
#Test Q2
    print("\nTest Q2")
    data=pd.read_csv("qa.csv", header=0)
#     print(data.head(52))
    doc_id=15
    x,y=find_similar_doc(doc_id, data["question"].values.tolist())
    print(x,y)
    print(data["question"].iloc[doc_id])
    print(data["question"].iloc[x])
    
    doc_id=51
    x,y=find_similar_doc(doc_id, data["question"].values.tolist())
    print(x,y)
    print(data["question"].iloc[doc_id])
    print(data["question"].iloc[x])

