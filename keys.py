# -*- coding: utf-8 -*-
"""
Created on Fri May 22 15:21:09 2020

@author: ishita
"""

import numpy as np
import pandas as pd
import twitter 
import mine_tweets
import textacy as tex
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from textacy import preprocessing
import sys

class twitter_analysis:
    def __init__(self,api, user1, user2,text = "",general = False):
        self.api = api
        self.user1 = user1
        self.user2 = user2
        self.text = text
        self.general = general
    def processing(self):
        mine   = mine_tweets.twitter_mine(self.api,result_limit = 150)
        hillary = mine.mine_user_tweets(user = self.user1)
        trump  = mine.mine_user_tweets(user = self.user2)
        df_hillary = pd.DataFrame(hillary)
        df_trump = pd.DataFrame(trump)
        tweets = pd.concat([df_trump,df_hillary], axis = 0)
        tweet_text = tweets['text'].values
        clean_text = [tex.preprocessing.replace_urls(x,"") for x in tweet_text]
        clean_text = [tex.preprocessing.replace_emails(x,"") for x in clean_text]
        clean_text = [tex.preprocessing.replace_hashtags(x,"") for x in clean_text]
        clean_text = [tex.preprocessing.replace_emojis(x,"") for x in clean_text]
        clean_text = [tex.preprocessing.replace_currency_symbols(x,"") for x in clean_text]
        clean_text = [tex.preprocessing.replace_phone_numbers(x,"") for x in clean_text]
        clean_text = [tex.preprocessing.replace_user_handles(x,"") for x in clean_text]
        clean_text = [tex.preprocessing.normalize_unicode(x) for x in clean_text]
        clean_text = [tex.preprocessing.normalize_hyphenated_words(x) for x in clean_text]
        clean_text = [tex.preprocessing.normalize_whitespace(x) for x in clean_text]
        target = tweets['handle'].map(lambda x: 1 if x == self.user1 else 0).values
        '''
        Using n_gram.py to find the n-grams. The n-grams found are similar to the one found 
        by the TfidVectorizer. To vectorize the data I will be using count vectorizer. The 
        run time for the file is 7 minutes.
        '''
        #clean_text = "".join(clean_text)
        #vect = n_gram.ngram(clean_text,2,4)
        #vect1 = vect.ngram_words()
        #count_vect = CountVectorizer(tokenizer=lambda text: text)
        #X_train_counts = count_vect.fit_transform(vect1)
        #print(X_train_counts.shape)
        
        tfv = TfidfVectorizer(ngram_range=(2,5), max_features=2000,stop_words = "english")
        X = tfv.fit_transform(clean_text).todense()
        '''
        Using logistic_regression.py.
        '''
        #reg = logistic_regression.logistic_regression(X,target,50000,5e-5)
        #reg = reg.regression()
        lr = LogisticRegression()
        params = {'penalty': ['l1', 'l2'], 'C':np.logspace(-5,0,100)}
        gs = GridSearchCV(lr, param_grid=params, cv=10, verbose=1)
        gs.fit(X, target)
        estimator = LogisticRegression(penalty='l2',C=1.0)
        estimator.fit(X,target)
        Xtest = tfv.transform(self.test)
        if self.general:
            Probas_x = pd.DataFrame(estimator.predict_proba(X), columns=["Proba_User1", "Proba_User2"])
            joined_x = pd.merge(tweets, Probas_x, left_index=True, right_index=True)
            joined_hillary = joined_x[joined_x['handle']== self.user2]
            for el in joined_hillary[joined_hillary['Proba_User2']==max(joined_hillary['Proba_User2'])]['text']:
                print ("most likely by User2:" , el)
            for el in joined_hillary[joined_hillary['Proba_User2']==min(joined_hillary['Proba_Hillary'])]['text']:
               print ("least likely by User2:" , el)
            joined_donald = joined_x[joined_x['handle']=="realDonaldTrump"]
            for el in joined_donald[joined_donald['Proba_User1']==max(joined_donald['Proba_User1'])]['text']:
                print ("most likely by User1:" , el)
            for el in joined_donald[joined_donald['Proba_User1']==min(joined_donald['Proba_User1'])]['text']:
               print ("least likely by User1:" , el)
        return pd.DataFrame(estimator.predict_proba(Xtest), columns=["Proba_User1", "Proba_User2"])
if __name__ == "__main__":
    consumer_key = sys.argv[1]
    consumer_secret = sys.argv[2]
    access_token_key = sys.argv[3]
    access_token_secret = sys.argv[4]
    user1 = sys.argv[5]
    user2 = sys.argv[6]
    text = sys.argv[7]
    general = sys.argv[8]
    
    api = twitter.Api(
        consumer_key         =   [consumer_key],
        consumer_secret      =   [consumer_secret],
        access_token_key     =   [access_token_key],
        access_token_secret  =   [access_token_secret],
        tweet_mode = 'extended'
    )
    
    tweet = twitter_analysis(api,user1,user2,text,general)
    print(tweet)
    
    




