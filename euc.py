
import sys
import datetime
import numpy
from nltk.cluster import KMeansClusterer, GAAClusterer, euclidean_distance
import nltk.corpus
from nltk import decorators
import nltk.stem
from sklearn.cluster import KMeans
import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans2, whiten
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
def calc_date(st):
    today=datetime.date.today()
    a=(st.split('T',2))[0].split('-',3)
    b=[]
    for i in a:
        b.append(int(i))
    calc=datetime.date(b[0],b[1],b[2])
    d=today-calc
    return d.days
stemmer_func = nltk.stem.snowball.EnglishStemmer().stem
stopwords = set(nltk.corpus.stopwords.words('english'))

@decorators.memoize
def normalize_word(word):
    return stemmer_func(word.lower())

def get_words(titles):
    words = set()
    for title in job_titles:
        for word in title.split():
            words.add(normalize_word(word))
    return list(words)

@decorators.memoize
def vectorspaced(title):
    title_components = [normalize_word(word) for word in title.split()]
    return numpy.array([
        word in title_components and not word in stopwords
        for word in words], numpy.short)
tim=[]
name=[]
day=[]
sqft=[]

lat_long=[]

dic={}

c=0
with open('Purpleyo_Items.csv','rb') as csvfile:
    reader=csv.DictReader(csvfile,)
#created 4 lists 
    for row in reader:
        tim.append(row['listing_date'])
        name.append(row['building_name'])
        lat_long.append([row['lat'],row['lng']])
        sqft.append(row['sqft'])

#calculated days passed for each data point
    for i in tim:
        day.append(calc_date(i))


    days=np.array(day)

    d=KMeans(n_clusters=3)
    d.fit(days.reshape(-1,1))

    a=int(d.n_clusters)
    i=0
    j=0
#initialize dictionary
    while i<a:
        dic.update({i:[]})
        i+=1
   # print dic
#fill dictionary with clustered values    
    g=0
    for i in day:
        b=d.predict(i)
        b=int(b)
        dic[b].append([tim[g],lat_long[g],name[g],sqft[g]])
        g+=1

kmeans2_n_clusters=5
#initialie global dictionary    
global_dic={}
i=0
while(i<(a*kmeans2_n_clusters)):
  global_dic.update({i:[]})
  i+=1


count=0

for i in dic:
    cluster_latlong_list=[]
    c=0
    #created temporary list within Each Cluster for Lat-long to be provided for Kmeans2
    for j in dic[i]:
        cluster_latlong_list.append([float(j[1][0]),float(j[1][1])])

   # print cluster_latlong_list

    cluster_latlong_numpy=np.array(cluster_latlong_list)
    
    #x = normalized data (lat,long) , y=corresponding label (cluster to which the lat long belongs)
    x,y = kmeans2(whiten(cluster_latlong_numpy), kmeans2_n_clusters, iter = 20) 
    
    u=0
    for u in y:
        k=int(u)
        global_dic[k+count].append(dic[i][c])
        c+=1
    count+=5

for x in list(global_dic.keys()):
    if global_dic[x]==[]:
        del global_dic[x]  
for k in global_dic:
    print global_dic[k]
    print "      "

c=1
count=0
for i in global_dic:
	job_titles=[]
	for j in global_dic[i]:
		if re.search('[0-9a-zA-Z0-9]',j[2]):
			job_titles.append(str(j[2]))
	print "yaha"
	print job_titles
	words = get_words(job_titles)
	#print "words yaha"
	#print words
	ks = range(1,10)
	if len(job_titles)>2:
		cluster = KMeansClusterer((int(len(job_titles)/3)+1), euclidean_distance,avoid_empty_clusters=True)
		
	else:
		cluster = KMeansClusterer(1, euclidean_distance,avoid_empty_clusters=True)
		#cluster = GAAClusterer(5)  
	cluster.cluster([vectorspaced(title) for title in job_titles if title])
    # called when you are classifying previously unseen examples!
	classified_examples = [
    cluster.classify(vectorspaced(title)) for title in job_titles
    ]
	for cluster_id, title in sorted(zip(classified_examples, job_titles)):
		print cluster_id, title[0]