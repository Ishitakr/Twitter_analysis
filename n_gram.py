# -*- coding: utf-8 -*-
"""
Created on Fri May 22 22:46:17 2020

@author: ishita
"""


import nltk
nltk.download('punkt')
from nltk.corpus import stopwords 
nltk.download('stopwords')
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from itertools import chain
import re

#from sklearn.feature_extraction.text import TfidfVectorizer


class ngram:
    def __init__(self,text,min_number = 1, max_number = 1):
        self.text = text
        self.min_number = min_number
        self.max_number = max_number
        
    def ngram_words(self):
        text = self.text
        prased_text = re.sub(r'[^A-Za-z. ]', '',text)
        words_token = nltk.word_tokenize(prased_text)
        tokens_without_sw = [word for word in words_token if not word in stopwords.words() and word !="see"]
        filtered_sentence = (" ").join(tokens_without_sw)
        words_token = nltk.word_tokenize(filtered_sentence)
        ngrams = []
        ngram2 = []
        words = self.min_number
        #print(words)
        while words <= self.max_number:
            for i in range(len(words_token) - words):
                seq = ' '.join(words_token[i: i + words])
                #print(seq)
                ngram2.append(seq)
            words +=1
            #print(type(ngram2))
        ngrams.append(ngram2)
        ngrams = chain.from_iterable(ngrams)
        return list(ngrams)



#text = "Today, we will study the N-Grams approach and will see how the N-Grams approach can be used to create a simple automatic text filler or suggestion engine. Automatic text filler is a very useful application and is widely used by Google and different smartphones where a user  and the remaining text is automatically populated or suggested by the application."
#test = ngram(text,1,2)
#test1 = test.ngram_words()
##print(test1)
#count_vect = CountVectorizer(tokenizer=lambda text: text)
#X_train_counts = count_vect.fit_transform(test1)
#print(test1)
#print(X_train_counts.shape)
#
##print(test1.shape)
#
##print(count)
#
###print(type(test1))
##count = Counter(words)
##print(count)
##df = pd.DataFrame(text)
#tfv = TfidfVectorizer(ngram_range=(1,2), stop_words='english')
#X = tfv.fit_transform(test1)
#print(tfv.get_feature_names())
##print (X.shape)

#
## Pulls all of trumps tweet text's into one giant string
##summaries = "".join(df)
#ngrams_summaries = vect.build_analyzer()(text)
#print(ngrams_summaries)