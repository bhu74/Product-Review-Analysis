#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
from pymongo import MongoClient
from NutritionUtils import *
import NutritionConfig as cfg

nltk.download('vader_lexicon')

client = MongoClient(host=cfg.DB_HOST, maxPoolSize=50)
collection = client[cfg.DB_NAME][cfg.COLLECTION]
data = pd.DataFrame(list(collection.find({}, {'_id':1, 'upc':1, 'reviews':1})))

review_details = []
topics_df = pd.DataFrame()
for idx, product in data.iterrows():
    print(idx, product['_id'])
    product_reviews = update_sentiment_scores(product)
    topics_df = get_topics(product_reviews)
    if not topics_df.empty:
        s1 = pd.pivot_table(topics_df, values=['Dominant_Topic'], index=['Topic_Keywords'], columns=['sentiment'], aggfunc='count').reset_index()
        s1.columns=s1.columns.droplevel(level=0)
        pos_list, neg_list = group_topics(s1)
        avg_pos = np.mean(topics_df.pos)
        avg_neu = np.mean(topics_df.neu)
        avg_neg = np.mean(topics_df.neg)
        collection.update({ '_id': ObjectId(product['_id'])},
                          { '$set': {'sentiment.positive': avg_pos, 'sentiment.neutral': avg_neu, 'sentiment.negative': avg_neg,
                                 'topics.positives': pos_list, 'topics.negatives': neg_list}}, upsert=True)
