# -*- coding: utf-8 -*-
import string
import collections
import csv 
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from pprint import pprint
import re
# import sys
# reload(sys)
# sys.setdefaultencoding("utf-8")
def process_text(text, stem=True):
    """ Tokenize text and stem words removing punctuation """
    text = text.encode('utf-8').translate(None, string.punctuation)
    tokens = word_tokenize(text)
 
    if stem:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(t) for t in tokens]
 
    return tokens
 
 
def cluster_texts(texts, clusters=3):
    """ Transform texts to Tf-Idf coordinates and cluster texts using K-Means """
    vectorizer = TfidfVectorizer(tokenizer=process_text,
                                 stop_words=stopwords.words('english'),
                                 max_df=2,
                                 min_df=0,
                                 lowercase=True)
    print "hii" 
    print texts
    tfidf_model = vectorizer.fit_transform(texts)
    km_model = KMeans(n_clusters=clusters,max_iter=500)
    km_model.fit(tfidf_model)
    #labels = km_model.predict(tfidf_model)
    # print "check"
    # for items in labels:
    # 	print items
    # order_centroids = km_model.cluster_centers_.argsort()[:, ::-1]
    #terms = vectorizer.get_feature_names()
    # for i in range(clusters):
    #     print "Cluster %d:" % i
    #     for ind in order_centroids[i, :250]:
    #         print " %s" % terms[ind]
    # print "termns here"
    # print terms
    clustering = collections.defaultdict(list)
    for idx, label in enumerate(km_model.labels_):
        clustering[label].append(idx)
 
    return clustering
 
 
if __name__ == "__main__":
    with open('Jagaha_Items.csv', 'rb') as f:
        reader = csv.reader(f)
        your_list = list(reader)
    v=[]
    c=1
    count=0
    for i in your_list:
        # if(c==1000):
        #     break
        # else:
        if re.search('[0-9a-zA-Z0-9]',i[4]):
            v.append(str(i[4]))
            c=c+1
    for j in v:
        print(j)
        count=count+1
    print count
    clusters = cluster_texts(v, 2000)
    pprint(dict(clusters))
