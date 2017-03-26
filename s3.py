

# coding: utf-8
from __future__ import print_function
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.cluster import KMeansClusterer, GAAClusterer, euclidean_distance
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans2, whiten
import sys,time,glob
import nltk.corpus
from nltk import decorators
import nltk.stem
import numpy as np
import pandas as pd
import nltk
import re,csv,boto3,botocore
import os
import codecs
from sklearn import feature_extraction
import mpld3
from nltk.stem.snowball import SnowballStemmer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.externals import joblib
from pyspark import SparkContext,SparkConf

# APP_NAME="myapp"
# conf = SparkConf().setAppName(APP_NAME)
# conf = conf.setMaster("local[*]")
# sc   = SparkContext(conf=conf)
# print("hiiiiiiiiiiiii")
# sc = SparkContext(conf = conf)
# s3 = boto3.resource('s3')
# bucket = s3.Bucket('oyeok-s3')
# exists = True
# try:
#     s3.meta.client.head_bucket(Bucket='mybucket')
# except botocore.exceptions.ClientError as e:
#     # If a client error is thrown, then check that it was a 404 error.
#     # If it was a 404 error, then the bucket does not exist.
#     error_code = int(e.response['Error']['Code'])
#     if error_code == 404:
#         exists = False
# for object in bucket.objects.all():
#     print(object)
# print("hihihiihih")
with open('/home/prajwal/Desktop/combinations/item_28000.csv', 'rb') as f:
        reader = csv.reader(f)
        next(reader,None)
        your_list = list(reader)
        job_titles=[]
        print('-----------------------------Stage 1----------------------------------------------------------------')
        print (' ')
        clust_list=[]
        c=1
        count=0
        ccount=0
        t=0 
        x=""
        i=0 
        films={}
        date_format = "%m/%d/%Y"
        config_type=[]
        lat=[]
        longi=[]
        buy_price=[]
        lease_price=[]
        buy_lease_price=[]
        sqfeet=[]
        listdate=[]
        txn_type=[]
        locality=[]
        city=[]
        prop=[]

        for i in your_list:
            if(i[1]!=""):
                if re.search('[0-9a-zA-Z0-9]',i[1]):
                    f=""
                    z=""
                    buy_price.append(str(i[3]))
                    lease_price.append(str(i[4]))
                    # if re.search('[0-9a-zA-Z0-9][\-]',i[1]):
                    #   print("before")
                    #   print(i[1])
                    #   i[1]=i[1].replace("-","")
                    #   print(i[1])
                    i[1]=i[1].decode('ascii','ignore')
                    job_titles.append(str(i[1]))
                    config_type.append(str(i[2]))
                    y=str(i[13])
                    y=y.replace(',','')
                    sqfeet.append(str(y))
                    lat.append(str(i[5]))
                    txn_type.append(str(i[10]))
                    longi.append(str(i[6]))
                    locality.append(str(i[12]))
                    city.append(str(i[8]))
                    f=str(i[0])+"-"+str(i[7])
                    prop.append(str(f))
                    today=time.strftime("%m/%d/%Y")
                    #print (today)
                    today = datetime.strptime(today, date_format)
                    #todaydate = datetime.date.today()
                    someday = datetime.strptime(str(i[9]), "%m/%d/%Y %H:%M:%S")
                    someday1 = someday.strftime(date_format)
                    someday1=datetime.strptime(someday1,date_format)
                    #print (someday)
                    diff = today-someday1
                    if(diff.days>7):   
                        listdate.append("Last month")
                    else:
                        listdate.append("Last week")
 
            else:
                job_titles.append("Empty")
    	        ccount=ccount+1

kmeans2_n_clusters=6
kmeans2_n_clusters_latlong=20
str_no_cluster=300                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
stopwords = nltk.corpus.stopwords.words('english')
stemmer = SnowballStemmer("english")
print("here is config")

print (len(config_type))
print(len(job_titles))
print(len(lat))
print(len(longi))
print(len(buy_price))
print(len(lease_price))
print(len(sqfeet))
print(len(txn_type))

def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.decode('ascii', 'ignore') for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens

stage_1_final_dic={}

totalvocab_stemmed = []
totalvocab_tokenized = []
for i in job_titles:
    allwords_stemmed = tokenize_and_stem(i) #for each item in 'synopses', tokenize/stem
    totalvocab_stemmed.extend(allwords_stemmed) #extend the 'totalvocab_stemmed' list
    
    allwords_tokenized = tokenize_only(i)
    totalvocab_tokenized.extend(allwords_tokenized)

vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)


tfidf_vectorizer = TfidfVectorizer(max_df=0.9999999999999999999999999999999999999999, max_features=200000,
                                 min_df=0.000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))
# for k in job_titles:
#   k=k.encode("utf-8").decode("utf-8")
tfidf_matrix = tfidf_vectorizer.fit_transform(job_titles) #fit the vectorizer to synopses

terms = tfidf_vectorizer.get_feature_names()

#dist = 1 - cosine_similarity(tfidf_matrix)






km = KMeans(n_clusters=str_no_cluster)

km.fit(tfidf_matrix)
joblib.dump(km,  'doc_cluster.pkl')
km = joblib.load('doc_cluster.pkl')
clusters = km.labels_.tolist()

films1={'Title':job_titles,'cluster':clusters,'Config':config_type,'Latitude':lat,'Longitude':longi,'BuyPrice':buy_price,'LeasePrice':lease_price,'Area':sqfeet,'ListDate':listdate,'TxnType':txn_type,"Locality":locality,"City":city,"Property":prop}
frame = pd.DataFrame(films1, index = [clusters], columns=['Title','Config','cluster','Latitude','Longitude','Price','BuyPrice','LeasePrice','Area','ListDate','TxnType','Locality','City','Property'])
k=0
i=0
while(i<(str_no_cluster)):
	stage_1_final_dic.update({i:[]})
	i+=1

strng=" "
strng1=""
stage_1_final_dic_counter=0
for i in range(str_no_cluster):
        strng=""
        strng1=""
        strng2=""
        strng3=""
        strng5=""
        strng7=""
        strng6=""
        strng4=""
        strng7=""
        strng8=""
        strng9=""
        strng10=""
        strng11=""

        try:
            for title in frame.ix[i]['Title']:
                if(len(title)==1):
                    strng+=str(title)
                    flag=1
                else:
                    flag=0
            for config in frame.ix[i]['Config']:
                if(len(config)==1):
                    strng1+=str(config)
                    flag1=1
                else:
                    flag1=0
            for lati in frame.ix[i]['Latitude']:
                if(len(lati)==1):
                    strng2+=str(lati)
                    flag2=1
                else:
                    flag2=0
            for longi in frame.ix[i]['Longitude']:
                if(len(longi)==1):
                    strng3+=str(longi)
                    flag3=1
                else:
                    flag3=0
            for bprices in frame.ix[i]['BuyPrice']:
                if(len(bprices)==1):
                    strng4+=str(bprices)
                    flag4=1
                else:
                    flag4=0
            for lprices in frame.ix[i]['LeasePrice']:
                if(len(lprices)==1):
                    strng5+=str(lprices)
                    flag5=1
                else:
                    flag5=0
            for area in frame.ix[i]['Area']:
                if(len(area)==1):
                    strng6+=str(area)
                    flag6=1
                else:
                    flag6=0
            for listdates in frame.ix[i]['ListDate']:
                if(len(listdates)==1):
                    strng7+=str(listdates)
                    flag7=1
                else:
                    flag7=0
            for txn_types in frame.ix[i]['TxnType']:
                if(len(txn_types)==1):
                    strng8+=str(txn_types)
                    flag8=1
                else:
                    flag8=0
            for loc in frame.ix[i]['Locality']:
                if(len(loc)==1):
                    strng9+=str(loc)
                    flag9=1
                else:
                    flag9=0
            for city in frame.ix[i]['City']:
                if(len(city)==1):
                    strng10+=str(city)
                    flag10=1
                else:
                    flag10=0
            for props in frame.ix[i]['Property']:
                if(len(props)==1):
                    strng11+=str(props)
                    flag11=1
                else:
                    flag11=0
            if (flag and flag1 and flag2 and flag3 and flag4 and flag5 and flag6 and flag7 and flag8 and flag9 and flag11 and flag10):
                stage_1_final_dic[stage_1_final_dic_counter].append([strng,strng1,strng6,strng4,strng5,strng2,strng3,strng7,strng8,strng9,strng10,strng11])
                stage_1_final_dic_counter+=1

			
	except KeyError:
		break
for i in range(str_no_cluster):
	strng=""
	strng1=""
	try:
		for title,config,lati,longitude,bprices,lprices,area,listdates,txn_types,loc,city,props in zip(frame.ix[i]['Title'],frame.ix[i]['Config'],frame.ix[i]['Latitude'],frame.ix[i]['Longitude'],frame.ix[i]['BuyPrice'],frame.ix[i]['LeasePrice'],frame.ix[i]['Area'],frame.ix[i]['ListDate'],frame.ix[i]['TxnType'],frame.ix[i]['Locality'],frame.ix[i]['City'],frame.ix[i]['Property']):
			
			if(len(title)!=1):
				stage_1_final_dic[stage_1_final_dic_counter].append([title,config,area,bprices,lprices,lati,longitude,listdates,txn_types,loc,city,props])
		stage_1_final_dic_counter+=1
	except KeyError:
		break
for i in stage_1_final_dic:
	print (stage_1_final_dic[i])
	print (" ")
	print (" ")

for x in list(stage_1_final_dic.keys()):
    if stage_1_final_dic[x]==[]:
        del stage_1_final_dic[x]

cleaned_names_stage_2_dic={}  
keys=0
new_key=0

for key,value in stage_1_final_dic.items():
    new_key=keys
    cleaned_names_stage_2_dic[new_key]=stage_1_final_dic[key]
    keys=keys+1
print("Cleaned name cluster**************************************************************************************************")
for i in cleaned_names_stage_2_dic:
    print (cleaned_names_stage_2_dic[i])
    print (" ")
    print (" ")
stage_1_final_dic.clear()
print("  ************************* bhk clustering *********************** ")
global_dic={}
i=0

commercial_dic={}
commercial_dic_counter=0
stage_2_final_dic={}
stage_2_final_dic_counter=0
while(i<(str_no_cluster*kmeans2_n_clusters)):
  commercial_dic.update({i:[]})
  i+=1  
i=0
while(i<(str_no_cluster*kmeans2_n_clusters)):
  stage_2_final_dic.update({i:[]})
  i+=1  
for i in cleaned_names_stage_2_dic:
    length=0
    cluster_bhk_list=[]
    config_type=[]
    lat=[]
    longi=[]
    buy_price=[]
    lease_price=[]
    buy_lease_price=[]
    sqfeet=[]
    listdate=[]
    txn_type=[]
    job_titles=[]
    locality=[]
    city=[]
    prop=[]
    c=0
#created temporary list within Each Cluster for Lat-long to be provided for Kmeans2
    for j in cleaned_names_stage_2_dic[i]:
        
        if(str(j[1])!="None"):
           
            bhk_str=j[1]
            job_titles.append(str(j[0]))
            cluster_bhk_list.append(str(bhk_str))
            sqfeet.append(str(j[2]))
            buy_price.append(str(j[3]))
            lease_price.append(str(j[4]))
            lat.append(str(j[5]))
            longi.append(str(j[6]))
            listdate.append(str(j[7]))
            txn_type.append(str(j[8]))
            locality.append(str(j[9]))
            city.append(str(j[10]))
            prop.append(str(j[11]))
            length+=1
        else:
            
            commercial_dic[commercial_dic_counter].append([str(j[0]),str(j[1]),str(j[2]),str(j[3]),str(j[4]),str(j[6]),str(j[5]),str(j[7]),str(j[8]),str(j[9]),str(j[10]),str(j[11])])
    commercial_dic_counter+=1
     
    if(len(cluster_bhk_list)!=0):
        totalvocab_stemmed = []
        totalvocab_tokenized = []
        for i in cluster_bhk_list:
            allwords_stemmed = tokenize_and_stem(i) #for each item in 'synopses', tokenize/stem
            totalvocab_stemmed.extend(allwords_stemmed) #extend the 'totalvocab_stemmed' list
            
            allwords_tokenized = tokenize_only(i)
            totalvocab_tokenized.extend(allwords_tokenized)

        vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)

    
        tfidf_vectorizer = TfidfVectorizer(max_df=0.999999999999999999999999999999999, max_features=200000,
                                         min_df=0.000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001, stop_words='english',
                                         use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))
        tfidf_matrix = tfidf_vectorizer.fit_transform(cluster_bhk_list) #fit the vectorizer to synopses

       
        terms = tfidf_vectorizer.get_feature_names()

        #dist = 1 - cosine_similarity(tfidf_matrix)




        if(length<6):
            kmeans2_n_clusters=length
            km = KMeans(n_clusters=kmeans2_n_clusters)    
        else:
            kmeans2_n_clusters=6
            km = KMeans(n_clusters=kmeans2_n_clusters)
       
        km.fit(tfidf_matrix)
        clusters = km.labels_.tolist()
        films2={'Title':job_titles,'cluster':clusters,'Config':cluster_bhk_list,'Latitude':lat,'Longitude':longi,'BuyPrice':buy_price,'LeasePrice':lease_price,'Area':sqfeet,'ListDate':listdate,'TxnType':txn_type,"Locality":locality,"City":city,"Property":prop}
        frame = pd.DataFrame(films2, index = [clusters], columns=['Title','Config','cluster','Latitude','Longitude','Price','BuyPrice','LeasePrice','Area','ListDate','TxnType','Locality','City','Property'])
        k=0
        i=0
        strng=" "
        strng1=""
        
        for i in range(kmeans2_n_clusters):
                strng=""
                strng1=""
                strng2=""
                strng3=""
                strng5=""
                strng7=""
                strng6=""
                strng4=""
                strng7=""
                strng8=""
                strng9=""
                strng10=""
                strng11=""

                try:
                    for title in frame.ix[i]['Title']:
                        if(len(title)==1):
                            strng+=str(title)
                            flag=1
                        else:
                            flag=0
                    for config in frame.ix[i]['Config']:
                        if(len(config)==1):
                            strng1+=str(config)
                            flag1=1
                        else:
                            flag1=0
                    for lati in frame.ix[i]['Latitude']:
                        if(len(lati)==1):
                            strng2+=str(lati)
                            flag2=1
                        else:
                            flag2=0
                    for longi in frame.ix[i]['Longitude']:
                        if(len(longi)==1):
                            strng3+=str(longi)
                            flag3=1
                        else:
                            flag3=0
                    for bprices in frame.ix[i]['BuyPrice']:
                        if(len(bprices)==1):
                            strng4+=str(bprices)
                            flag4=1
                        else:
                            flag4=0
                    for lprices in frame.ix[i]['LeasePrice']:
                        if(len(lprices)==1):
                            strng5+=str(lprices)
                            flag5=1
                        else:
                            flag5=0
                    for area in frame.ix[i]['Area']:
                        if(len(area)==1):
                            strng6+=str(area)
                            flag6=1
                        else:
                            flag6=0
                    for listdates in frame.ix[i]['ListDate']:
                        if(len(listdates)==1):
                            strng7+=str(listdates)
                            flag7=1
                        else:
                            flag7=0
                    for txn_types in frame.ix[i]['TxnType']:
                        if(len(txn_types)==1):
                            strng8+=str(txn_types)
                            flag8=1
                        else:
                            flag8=0
                    for loc in frame.ix[i]['Locality']:
                        if(len(loc)==1):
                            strng9+=str(loc)
                            flag9=1
                        else:
                            flag9=0
                    for city in frame.ix[i]['City']:
                        if(len(city)==1):
                            strng10+=str(city)
                            flag10=1
                        else:
                            flag10=0
                    for props in frame.ix[i]['Property']:
                        if(len(props)==1):
                            strng11+=str(props)
                            flag11=1
                        else:
                            flag11=0

                    if (flag and flag1 and flag2 and flag3 and flag4 and flag5 and flag6 and flag7 and flag8 and flag9 and flag11 and flag10):

                        stage_2_final_dic[stage_2_final_dic_counter].append([strng,strng1,strng6,strng4,strng5,strng2,strng3,strng7,strng8,strng9,strng10,strng11])
                        stage_2_final_dic_counter+=1
                except KeyError:
                    break
        for i in range(kmeans2_n_clusters):
            strng=""
            strng1=""
         
            try:
                for title,config,lati,longitude,bprices,lprices,area,listdates,txn_types,loc,city,prop in zip(frame.ix[i]['Title'],frame.ix[i]['Config'],frame.ix[i]['Latitude'],frame.ix[i]['Longitude'],frame.ix[i]['BuyPrice'],frame.ix[i]['LeasePrice'],frame.ix[i]['Area'],frame.ix[i]['ListDate'],frame.ix[i]['TxnType'],frame.ix[i]['Locality'],frame.ix[i]['City'],frame.ix[i]['Property']):
                    if(len(title)!=1):
                        stage_2_final_dic[stage_2_final_dic_counter].append([title,config,area,bprices,lprices,lati,longitude,listdates,txn_types,loc,city,prop])
                stage_2_final_dic_counter+=1
            except KeyError:
                continue

    else:
        continue

print() 
print()
for x in list(commercial_dic.keys()):
    if commercial_dic[x]==[]:
        del commercial_dic[x]

for x in list(stage_2_final_dic.keys()):
    if stage_2_final_dic[x]==[]:
        del stage_2_final_dic[x]
cleaned_names_stage_3_dic={}  
i=0
while(i<(str_no_cluster*kmeans2_n_clusters)):
  cleaned_names_stage_3_dic.update({i:[]})
  i+=1  
keys=0
new_key=0

for key,value in stage_2_final_dic.items():
    new_key=keys
    cleaned_names_stage_3_dic[new_key]=stage_2_final_dic[key]
    keys=keys+1
for x in list(cleaned_names_stage_3_dic.keys()):
    if cleaned_names_stage_3_dic[x]==[]:
        del cleaned_names_stage_3_dic[x]
stage_2_final_dic.clear()
print("***************************Cleaned names bhk cluster**************************************************************************************************")
for f in cleaned_names_stage_3_dic:
    print (cleaned_names_stage_3_dic[f])
    print (" ")

kmeans2_n_clusters=6
if(len(cleaned_names_stage_3_dic)!=0):
    global_dic={}
    i=0
    while(i<(stage_2_final_dic_counter*kmeans2_n_clusters*kmeans2_n_clusters)):
      global_dic.update({i:[]}) 
      i+=1
    i=0
    count=0
    for i in cleaned_names_stage_3_dic:
        cluster_bhk_list=[]
        c=0
    
        for j in cleaned_names_stage_3_dic[i]:
            bhk_str=j[1]

            if not re.search('[0-9]',bhk_str[:1]):
                j[1]="2BHK"

            bhk_str=j[1]    
            new_bhk=bhk_str[:1]

            a=float(new_bhk)

            cluster_bhk_list.append(a)
        cluster_bhk_numpy=np.array(cluster_bhk_list)
        # print("BHK list is")
        # print(cluster_bhk_list)
    #x = normalized data (lat,long) , y=corresponding label (cluster to which the lat long belongs)
        # print ' x and y'
        if len(cluster_bhk_list)>1:
            x,y = kmeans2(whiten(cluster_bhk_numpy), kmeans2_n_clusters, iter = 850) #450,550,haa 850,

            u=0
            # print("y is")
            # print(y)
            for u in y:
                k=int(u)

                try:
                    
                    global_dic[k+count].append(cleaned_names_stage_3_dic[i][c])
                    c+=1
                    # print("not in except")
                except KeyError:
                    # print("in except")
                    for k in range(0,len(cleaned_names_stage_3_dic[i])):
                        global_dic[count].append(cleaned_names_stage_3_dic[i][k]) 
                    break;
                              
                
        else:
            global_dic[count].append(cleaned_names_stage_3_dic[i][c])
        count+=kmeans2_n_clusters  
        

    print("  --------------- bhk clusters------------------------")
        #initialie global dictionary    
    for x in list(global_dic.keys()):
        if global_dic[x]==[]:
            del global_dic[x]  

    new_global_dic={}
    trimmeddic={}
    i=0
    while(i<(len(global_dic))):
        trimmeddic.update({i:[]})
        i+=1
    keys=0
    new_key=0
    for key,value in global_dic.items():
        new_key=keys
        trimmeddic[new_key]=global_dic[key]
        keys=keys+1
    for k in trimmeddic:
        print( "  ")
        print("  --------------- cluster ------------------------")
        for j in trimmeddic[k]:
            print (j[0],j[1],j[2])
global_dic.clear()
keys=int(len(trimmeddic))

for key,value in commercial_dic.items():
    trimmeddic[keys]=commercial_dic[key]
    keys=keys+1
commercial_dic.clear()
i=0
while(i<(commercial_dic_counter*stage_2_final_dic_counter*kmeans2_n_clusters_latlong)):
    new_global_dic.update({i:[]})
    i+=1

count_bada=0

for i in trimmeddic:
    cluster_latlong_list=[]
    c=0

    for j in trimmeddic[i]:
        cluster_latlong_list.append([float(j[5]),float(j[6])])
    cluster_latlong_numpy=np.array(cluster_latlong_list)
    digits=cluster_latlong_numpy
    if len(cluster_latlong_list)>1:
        try:
            x,y = kmeans2(whiten(digits), kmeans2_n_clusters_latlong, iter = 550,minit='points')                 
        except:
            k=0;
            for k in range(0,int(len(cluster_latlong_list)/2)):
                cluster_latlong_list[k][0]=cluster_latlong_list[k][0]+0.0000000000000001
                cluster_latlong_list[k][1]=cluster_latlong_list[k][1]+0.0000000000000001
            #print("after ")
            #print(cluster_latlong_list)
            digits=np.array(cluster_latlong_list)
            #print np.all(np.linalg.eigvalsh(digits))
            x,y = kmeans2(whiten(digits), kmeans2_n_clusters_latlong, iter = 550,minit='points') 
        
        u=0
        for u in y:
            k=int(u)

            try:
                new_global_dic[k+count_bada].append(trimmeddic[i][c])
                c+=1
            except KeyError:
                for k in range(0,len(trimmeddic[i])):
                    new_global_dic[0+count_bada].append(trimmeddic[i][k]) 
                break;       
            
            
    else:
        new_global_dic[0+count_bada].append(trimmeddic[i][c])        
    count_bada+=kmeans2_n_clusters_latlong

final_trimmed_dic={}
for x in list(new_global_dic.keys()):
    if new_global_dic[x]==[]:
        del new_global_dic[x] 
keys=0
new_key=0
for key,value in new_global_dic.items():
    new_key=keys
    final_trimmed_dic[new_key]=new_global_dic[key]
    keys=keys+1
praj=0
print ("lat-long clustering(*****************************************************************")
for i in final_trimmed_dic:
    print (" count")
    print (" ")
    for j in final_trimmed_dic[i]:
        print (j[0],j[1],j[2],j[5],j[6],j[11])
        praj+=1
new_global_dic.clear()
with open('final_output_bhk.csv', 'wb') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames = ["Building Name", "Config","Area (sq ft)","BuyPrice","LeasePrice","Longitude","Latitude", "Listing time","Type","Locality","City","Property_id-Platform","No. of Listings"], delimiter = ';')
    writer.writeheader()
    spamwriter = csv.writer(csvfile, delimiter=',')
    for k in trimmeddic:
        if len(trimmeddic[k])==1:
            listing=1
            trimmeddic[k][0].append(listing)
            spamwriter.writerow(trimmeddic[k][0])
        else:
            avg_b=0.0
            avg_l=0.0
            number=0
            property_ids=""
            listing=len(trimmeddic[k])
            for t in range(0,len(trimmeddic[k])):
              if(trimmeddic[k][t][7]=='Last month'):
                  try:
                      avg_b+=float(trimmeddic[k][t][3])
                      avg_l+=float(trimmeddic[k][t][4])
                      property_ids=property_ids+"|"+trimmeddic[k][t][11]
                      number+=1
                      index=t
                  except ValueError:
                      avg_b+=0
                      avg_l+=0
            area=trimmeddic[k][index][2]
            name=trimmeddic[k][index][0]
            bprices=avg_b/number
            lprices=avg_l/number
            conf=trimmeddic[k][index][1]
            lat=trimmeddic[k][index][5]
            txn_types=trimmeddic[k][index][8]
            longi=trimmeddic[k][index][6]
            listdate=trimmeddic[k][index][7]
            loc=trimmeddic[k][index][9]
            city=trimmeddic[k][index][10]
            prop=property_ids#trimmeddic[k][index][11]
            spamwriter.writerow([name,conf,area,bprices,lprices,lat,longi,listdate,txn_types,loc,city,prop,listing])
with open('final_output_lat.csv', 'wb') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames = ["Longitude","Latitude","BuyPrice","LeasePrice", "Listing time","Type","Locality","City","Property_id-Platform","No. of Listings"], delimiter = ';')
    writer.writeheader()
    spamwriter = csv.writer(csvfile, delimiter='\t')
    
    for k in final_trimmed_dic:
        listing=len(final_trimmed_dic[k])
        if len(final_trimmed_dic[k])==1:
            bprices=final_trimmed_dic[k][0][3]
            lprices=final_trimmed_dic[k][0][4]
            lat=final_trimmed_dic[k][0][5]
            longi=final_trimmed_dic[k][0][6]
            listdate=final_trimmed_dic[k][0][7]
            txn_types=final_trimmed_dic[k][0][8]
            loc=final_trimmed_dic[k][0][9]
            city=final_trimmed_dic[k][0][10]
            prop=final_trimmed_dic[k][0][11]
            spamwriter.writerow([lat,longi,bprices,lprices,listdate,txn_types,loc,city,prop,listing])
        else:
            avg_b=0.0
            avg_l=0.0
            number=0
            property_ids=""
            listing=len(final_trimmed_dic[k])
            for t in range(0,len(final_trimmed_dic[k])):
              if(final_trimmed_dic[k][t][7]=='Last month'):
                  try:
                      avg_b+=float(final_trimmed_dic[k][t][3])
                      avg_l+=float(final_trimmed_dic[k][t][4])
                      property_ids=property_ids+"|"+final_trimmed_dic[k][t][11]
                      number+=1
                      index=t
                  except ValueError:
                      avg_b+=0
                      avg_l+=0 
                
            area=final_trimmed_dic[k][index][2]
            name=final_trimmed_dic[k][index][0]
            bprices=avg_b/number
            lprices=avg_l/number
            conf=final_trimmed_dic[k][index][1]
            lat=final_trimmed_dic[k][index][5]
            longi=final_trimmed_dic[k][index][6]
            listdate=final_trimmed_dic[k][index][7]
            txn_types=final_trimmed_dic[k][index][8]
            loc=final_trimmed_dic[k][index][9]
            city=final_trimmed_dic[k][index][10]
            prop=property_ids
            spamwriter.writerow([lat,longi,bprices,lprices,listdate,txn_types,loc,city,prop,listing])
           
    
