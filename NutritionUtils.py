#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import nltk
import gensim
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import sentiment
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.corpora import Dictionary
from gensim.models.phrases import Phrases, Phraser
from bson.objectid import ObjectId
from pymongo import MongoClient
import NutritionConfig as cfg
import warnings
warnings.filterwarnings("ignore")

def remove_punct(w_list):
    '''
    Remove punctuations from words list
    '''
    return [word for word in w_list if word.isalpha()]

def select_noun(w_list):
    '''
    Select noun words from words list
    '''
    return [word for (word, pos) in nltk.pos_tag(w_list) if pos[:2] in ['NN', 'NNS', 'NNP', 'NNPS']]

def get_words_list(sentence):
    '''
    Pre-processing of review comments, returns words list
    '''
    sentence = sentence.lower()
    words_list = word_tokenize(sentence)
    words_list = select_noun(words_list)
    stop = set(stopwords.words('english'))
    words_list = [word for word in words_list if word not in stop]
    words_list = remove_punct(words_list)
    lemma = WordNetLemmatizer()
    words_list = [lemma.lemmatize(word) for word in words_list]
    return list(set(words_list))

def update_sentiment_scores(product):
    '''
    Update sentiment scores for all reviews of a product
    '''
    client = MongoClient(host=cfg.DB_HOST, maxPoolSize=50)
    collection = client[cfg.DB_NAME][cfg.COLLECTION]
    review_list =  []
    score_dict = dict()
    sid = SentimentIntensityAnalyzer()
    for rev in product['reviews']:
        scores = sid.polarity_scores(rev['textContent'])
        if scores['compound'] >= 0.05:
            sent = 'positive'
        elif scores['compound'] <= -0.05:
            sent = 'negative'
        else:
            sent = 'neutral'
        collection.update({ '_id': ObjectId(product['_id']), 'reviews': { '$elemMatch': {'_id': ObjectId(rev['_id'])}} },
                          { '$set': {'reviews.$.sentiment.positive': scores['pos'], 'reviews.$.sentiment.neutral': scores['neu'], \
                                     'reviews.$.sentiment.negative': scores['neg'], 'reviews.$.sentiment.compound': scores['compound']}}, upsert=True)
        review_list.append({'Review_id': rev['_id'], 'review': rev['textContent'], \
                            'sentiment':sent, 'pos': scores['pos'], 'neu': scores['neu'], 'neg': scores['neg']})
    return review_list

def get_topics(product_reviews):
    '''
    Identify the topics discussed in the product reviews
    '''
    cols = ['Review_id', 'review', 'sentiment', 'pos', 'neu', 'neg']
    review_df = pd.DataFrame(product_reviews, columns = cols)
    review_df['review_words'] = review_df['review'].apply(lambda x: get_words_list(x))

    text_data = review_df['review_words'].tolist()
    vocab = Dictionary(text_data)
    corpus = [vocab.doc2bow(text) for text in text_data]
    try:
        ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = 5, id2word=vocab, passes=15)
        sent_topics_df = pd.DataFrame()
        for i, row in enumerate(ldamodel[corpus]):
            row = sorted(row, key=lambda x: x[1], reverse=True)
            for j, (topic_num, prop_topic) in enumerate(row):
                if j == 0:
                    wp = ldamodel.show_topic(topic_num, topn=3)
                    topic_keywords = "-".join([word for word, prop in wp])
                    sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), topic_keywords]), ignore_index=True)
                else:
                    break
        sent_topics_df.columns = ['Dominant_Topic', 'Topic_Keywords']
        review_df = pd.concat([review_df, sent_topics_df], axis=1)
        return (review_df)
    except:
        return pd.DataFrame()

def group_topics(prod_df):
    '''
    Get the list of topics for positive and negative sentiments
    '''
    prod_df.rename(columns = {'':'Topic_Keywords'}, inplace = True)
    cols = prod_df.columns
    if 'positive' not in cols:
        prod_df['positive'] = 0
    if 'negative' not in cols:
        prod_df['negative'] = 0
    if 'neutral' not in cols:
        prod_df['neutral'] = 0
    pos_list=[]
    neg_list=[]
    pos_min, pos_max = min(prod_df['positive']), max(prod_df['positive'])
    neg_min, neg_max = min(prod_df['negative']), max(prod_df['negative'])
    prod_df['pos_norm'] = prod_df['positive'].apply(lambda x: normalize_value(x, pos_min, pos_max))
    prod_df['neg_norm'] = prod_df['negative'].apply(lambda x: normalize_value(x, neg_min, neg_max))
    for i, r in prod_df.iterrows():
        if r['pos_norm'] >= r['neg_norm']:
            pos_list.append(r['Topic_Keywords'])
        else:
            neg_list.append(r['Topic_Keywords'])
    return pos_list, neg_list

def normalize_value(x, col_min, col_max):
    '''
    Normalize the value between 0 & 1
    '''
    try:
        return (x - col_min)/(col_max - col_min)
    except:
        return 0
