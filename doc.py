from __future__ import print_function

from nltk.cluster import KMeansClusterer, GAAClusterer, euclidean_distance
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans2, whiten
import sys,time
import nltk.corpus
from nltk import decorators
import nltk.stem
import numpy as np
import pandas as pd
import nltk
import re,csv
import os
import codecs
from sklearn import feature_extraction
import mpld3
from nltk.stem.snowball import SnowballStemmer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.externals import joblib


from sklearn.feature_extraction.text import TfidfVectorizer
#titles=['The Godfather', 'The Godfather', "Schindler's List", 'Raging Bull', 'Raging Bulls', 'One Flew Over the Cuckoos Nest','One Flew Over the Cuckoos Nest', 'Citizen Kane', 'Citizen Kane', 'The Wizard of Oz', 'Titanic']
with open('Jagaha_Items.csv', 'rb') as f:
        reader = csv.reader(f)
        next(reader,None)
        your_list = list(reader)

        str_no_cluster=100
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
        date_format = "%d/%m/%Y"
        config_type=[]
        lat=[]
        longi=[]
        buy_price=[]
        lease_price=[]
        buy_lease_price=[]
        sqfeet=[]
        listdate=[]
        txn_type=[]
        # while(i<len(your_list)):
        # 	films.update({i:[]})
        # 	i+=1
        for i in your_list:
            if(i[4]!=""):
                if re.search('[0-9a-zA-Z0-9]',i[4]):
                    if ("Lease" ==i[20]):
                        x = str(i[16]) 
                        x= x[4:-10]
                        x=x.replace(',','')
                        lease_price.append(str(x))
                        buy_price.append(str(0))
                        txn_type.append(str(i[20]))
                        t=t+1
                    elif ('Buy'==i[20]):
                        x =str(i[19])
                        x= x[4:]
                        x=x.replace(',','') 
                        # clust_list.append([str(i[4]),str(i[3]),float(x[4:]),float(i[12]),float(i[13])])
                        # config_type.append(str(i[3]))
                        buy_price.append(str(x))
                        lease_price.append(str(0))
                        txn_type.append(str(i[20]))
                        #films[t].append([str(i[4]),str(i[3]),float(x),float(i[12]),float(i[13])])
                        t=t+1
                    else:
                        x = str(i[16]) 
                        x= x[4:-10]
                        x=x.replace(',','')
                        lease_price.append(str(x))
                        x =str(i[19])
                        x= x[4:]
                        x=x.replace(',','') 
                        buy_price.append(str(x))
                        txn_type.append("Buy/Lease")
                    job_titles.append(str(i[4]))
                    config_type.append(str(i[3]))
                    y=str(i[2])
                    y=y.replace(',','')
                    sqfeet.append(str(y))
                    lat.append(str(i[12]))
                    longi.append(str(i[13]))
                    today=time.strftime("%d/%m/%Y")
                    #print (today)
                    today = datetime.strptime(today, date_format)
                    #todaydate = datetime.date.today()
                    someday = datetime.strptime(str(i[7]), date_format)
                    #print (someday)
                    diff = today-someday
                    if(diff.days>7):   
                        listdate.append("Last month")
                    else:
                        listdate.append("Last week")

            else:
                job_titles.append("Empty")
    	        ccount=ccount+1

kmeans2_n_clusters=4
kmeans2_n_clusters_latlong=20
str_no_cluster = 600
stopwords = nltk.corpus.stopwords.words('english')
stemmer = SnowballStemmer("english")
print("here is config")

# print (len(config_type))
# print(len(job_titles))
# print(len(lat))
# print(len(longi))
# print(len(price))
# print (lat)
def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
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
# print  (vocab_frame.shape[0]) 

# print (vocab_frame.head())
print
print

# tfidf_vectorizer = TfidfVectorizer(max_df=0.99, max_features=200000,
#                                  min_df=0.0000000000000000000000000000000001, stop_words='english',
#                                  use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))
tfidf_vectorizer = TfidfVectorizer(max_df=0.4, max_features=200000,
                                 min_df=0.00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))
tfidf_matrix = tfidf_vectorizer.fit_transform(job_titles) #fit the vectorizer to synopses

# print(tfidf_matrix.shape)
terms = tfidf_vectorizer.get_feature_names()

dist = 1 - cosine_similarity(tfidf_matrix)
print
print





km = KMeans(n_clusters=str_no_cluster)

km.fit(tfidf_matrix)
# a=["Title","Config"]
# films={e1:0 for e1 in a}

	

# keys = ["Title","Config","cluster"]
# somekey="Title"
# films={key: [] for key in keys}
# print (films)


#uncomment the below to save your model 
#since I've already run my model I am loading from the pickle

joblib.dump(km,  'doc_cluster.pkl')
km = joblib.load('doc_cluster.pkl')
clusters = km.labels_.tolist()
# # print ("clusters name")
# # print (clusters)
# # print(len(clusters))	
# # print("clusters ithe aahet")
# # print (clusters)

# for i in clust_list:
# 	films["Title"].append(i[0])
# 	films['Config'].append(i[1])
# print("idhar dekh")
# films["cluster"].append(clusters)

#print (clusters)
#print(films['Title'])
films1={'Title':job_titles,'cluster':clusters,'Config':config_type,'Latitude':lat,'Longitude':longi,'BuyPrice':buy_price,'LeasePrice':lease_price,'Area':sqfeet,'ListDate':listdate,'TxnType':txn_type}
frame = pd.DataFrame(films1, index = [clusters], columns=['Title','Config','cluster','Latitude','Longitude','Price','BuyPrice','LeasePrice','Area','ListDate','TxnType'])
# print("frame")
# print("yayayayayayyayayay")
# print (frame['title'].value_counts())
# print ("here")
# print (frame)

# print (frame.ix[0]['title'])
# print("Top terms per cluster:")
# print()
#sort cluster centers by proximity to centroid
#order_centroids = km.cluster_centers_.argsort()[:, ::-1] 
# print ("ithe bagh")
k=0
# while (frame.ix[k]['title']):
#dic={}
# 	k=k+1


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
            if (flag and flag1 and flag2 and flag3 and flag4 and flag5 and flag6 and flag7 and flag8):
                # print ('')
                # print (strng)
                # print ('')
                # print (strng1)
                # print ('')
                # print (strng2)
                # print ('')
                # print (strng3)
                # print ('')
                # print (strng4)
                stage_1_final_dic[stage_1_final_dic_counter].append([strng,strng1,strng6,strng4,strng5,strng2,strng3,strng7,strng8])
                stage_1_final_dic_counter+=1


			
	except KeyError:
		break
for i in range(str_no_cluster):
	strng=""
	strng1=""
	#print("Cluster %d titles:" % i)
	try:
		for title,config,lati,longitude,bprices,lprices,area,listdates,txn_types in zip(frame.ix[i]['Title'],frame.ix[i]['Config'],frame.ix[i]['Latitude'],frame.ix[i]['Longitude'],frame.ix[i]['BuyPrice'],frame.ix[i]['LeasePrice'],frame.ix[i]['Area'],frame.ix[i]['ListDate'],frame.ix[i]['TxnType']):
					
			# if(len(title)==1):
			# 			# print ("debug")
			# 	strng+=str(title)
			# 	strng1+=str(config)
			# 	flag=1
			# else:	
			# 	flag=0
			# 	print(' %s,'% title)
			# 	print(' %s,'% config)
			# 	print(' %s,'% lati)
			# 	print(' %s,'% longitude)
			# 	print(' %s,'% prices)
				# print(' %s,'% title[1])
					
					#dic[i].append([title])
		# if(flag==1):
			# print (strng)
		
			# print(strng1)
			if(len(title)!=1):
				# print(' %s,'% title)
				# print(' %s,'% config)
				# print(' %s,'% lati)
				# print(' %s,'% longitude)
				# print(' %s,'% prices)
				stage_1_final_dic[stage_1_final_dic_counter].append([title,config,area,bprices,lprices,lati,longitude,listdates,txn_types])
		stage_1_final_dic_counter+=1
	except KeyError:
		break
	print() #add whitespace
	print() #add whitespace
print() 
print()
print()
for i in stage_1_final_dic:
	print (stage_1_final_dic[i])
	print (" ")
	print (" ")

for x in list(stage_1_final_dic.keys()):
    if stage_1_final_dic[x]==[]:
        del stage_1_final_dic[x]

cleaned_names_dic={}  
keys=0
new_key=0

for key,value in stage_1_final_dic.items():
    new_key=keys
    cleaned_names_dic[new_key]=stage_1_final_dic[key]
    keys=keys+1

#print"  ************************* bhk clustering *********************** "
global_dic={}
i=0
while(i<(str_no_cluster*kmeans2_n_clusters)):
  global_dic.update({i:[]})
  i+=1

count=0

# for x in list(dic.keys()):
#     if dic[x]==[]:
#         del dic[x]  
for i in cleaned_names_dic:
    cluster_bhk_list=[]
    c=0
#created temporary list within Each Cluster for Lat-long to be provided for Kmeans2
    for j in cleaned_names_dic[i]:
        bhk_str=j[1]
        # try:
        #     new_bhk=bhk_str[:1]
        #     a=float(new_bhk)
        # except ValueError:
        #     a=1
        if not re.search('[0-9]',bhk_str[:1]):
            j[1]="1 BHK"

        bhk_str=j[1]    
        new_bhk=bhk_str[:1]

        a=float(new_bhk)

        cluster_bhk_list.append(a)

#    print cluster_latlong_list
#    print '     bhk list  length '
    # for pr in cluster_bhk_list:
    #     print pr
    # print '   '
 #   print len(cluster_bhk_list)
    cluster_bhk_numpy=np.array(cluster_bhk_list)
    
#x = normalized data (lat,long) , y=corresponding label (cluster to which the lat long belongs)
    # print ' x and y'
    if len(cluster_bhk_list)>1:
        x,y = kmeans2(whiten(cluster_bhk_numpy), kmeans2_n_clusters, iter = 20) 
     #   print 'after xy ------------------------'+str(kiti)
     
      #  print y[0]
#        kiti+=1
        u=0
        print("here")
        print (y)
        for u in y:
            k=int(u)
    #        print 'k cartoy print' + str(k)
            try:
                
                global_dic[k+count].append(cleaned_names_dic[i][c])
                c+=1
            except KeyError:
                global_dic[0+count].append(cleaned_names_dic[i][c])        
            
    else:
        global_dic[0+count].append(cleaned_names_dic[i][c])
    count+=kmeans2_n_clusters  
    

print("  --------------- bhk clusters------------------------")

    #initialie global dictionary    
for x in list(global_dic.keys()):
    if global_dic[x]==[]:
        del global_dic[x]  

new_global_dic={}

# for i in global_dic:
#     #print "  "
#     #print "  --------------- cluster "+str(i)+"------------------------"
#     for k in global_dic[i]:
#         print k[1]

trimmeddic={}
while(i<(len(global_dic))):
    trimmeddic.update({i:[]})
    i+=1

#print 'trimed -----------'
# for k in trimmeddic:
#     print trimmeddic[k]
# # for j in global_dic:
#     trimmeddic.append(global_dic[i]) 
#trimmeddic=global_dic.copy()

#for k in trimmeddic:
#     print trimmeddic[k]
keys=0
new_key=0
for key,value in global_dic.items():
    new_key=keys
    trimmeddic[new_key]=global_dic[key]
    keys=keys+1
#print "the fineal is herer########################################################"
# for k,v in trimmeddic.items():
#     print k
#     print v
#     print " "
#


for k in trimmeddic:
    print( "  ")
    print("  --------------- cluster ------------------------")
    for j in trimmeddic[k]:
        print (j[0],j[1])

# i=0
# while(i<(str_no_cluster*kmeans2_n_clusters_latlong*kmeans2_n_clusters)):
#     new_global_dic.update({i:[]})
#     i+=1

# count_bada=0

# for i in trimmeddic:
#     cluster_latlong_list=[]
#     c=0

#     for j in trimmeddic[i]:
#         cluster_latlong_list.append([float(j[5]),float(j[6])])
#     cluster_latlong_numpy=np.array(cluster_latlong_list)
#     digits=cluster_latlong_numpy
#     if len(cluster_latlong_list)>1:
#         try:
#             x,y = kmeans2(whiten(digits), kmeans2_n_clusters_latlong, iter = 20,minit='points')                 
#         except:
#             print ('in here ')
#             print (cluster_latlong_list)
#             k=0;
#             for k in range(0,int(len(cluster_latlong_list)/2)):
#                 cluster_latlong_list[k][0]=cluster_latlong_list[k][0]+0.0000000000000001
#                 cluster_latlong_list[k][1]=cluster_latlong_list[k][1]+0.0000000000000001
#             print("after ")
#             print(cluster_latlong_list)
#             digits=np.array(cluster_latlong_list)
#             #print np.all(np.linalg.eigvalsh(digits))
#             x,y = kmeans2(whiten(digits), kmeans2_n_clusters_latlong, iter = 20,minit='points') 
#         print("y is here")
#         print (y)
#         u=0
#         for u in y:
#             k=int(u)
#    #         print 'k cartoy print' + str(k)
#             # try:
#             print ("k"+str(k))
#             print ("count"+str(count_bada))
#             print(i)
#             print (c)
#             #print(trimmeddic[i][c])
#             try:
#                 new_global_dic[k+count_bada].append(trimmeddic[i][c])
#                 c+=1
#             except KeyError:
#                 new_global_dic[0+count_bada].append(trimmeddic[i][c])        
            
            
#     else:
#         new_global_dic[0+count_bada].append(trimmeddic[i][c])        
#     count_bada+=kmeans2_n_clusters_latlong

# final_trimmed_dic={}
# for x in list(new_global_dic.keys()):
#     if new_global_dic[x]==[]:
#         del new_global_dic[x] 
# keys=0
# new_key=0
# for key,value in new_global_dic.items():
#     new_key=keys
#     final_trimmed_dic[new_key]=new_global_dic[key]
#     keys=keys+1
# print ("lat-long clustering(*****************************************************************")
# for i in final_trimmed_dic:
#     print (" count")
#     print (" ")
#     print (final_trimmed_dic[i])
with open('output_bhk.csv', 'wb') as csvfile:
    # spamwriter = csv.writer(csvfile, newline='')
    writer = csv.DictWriter(csvfile, fieldnames = ["Building Name", "Config","Area (sq ft)","BuyPrice","LeasePrice","Longitude","Latitude", "Listing time","Type","No. of Listings"], delimiter = ';')
    writer.writeheader()
    spamwriter = csv.writer(csvfile, delimiter='\t')
    for k in trimmeddic:
        if len(trimmeddic[k])==1:
            listing=1
            trimmeddic[k][0].append(listing)
            spamwriter.writerow(trimmeddic[k][0])
            
        elif trimmeddic[k][0][0]!=trimmeddic[k][1][0]:
            listing=1
            for t in range(0,len(trimmeddic[k])):
                trimmeddic[k][t].append(listing)
                spamwriter.writerow(trimmeddic[k][t])
               
        else:
            avg_b=0.0
            avg_l=0.0
            listing=len(trimmeddic[k])
            for t in range(0,len(trimmeddic[k])):
                try:
                    avg_b+=float(trimmeddic[k][t][3])
                    avg_l+=float(trimmeddic[k][t][4])
                except ValueError:
                    avg_b+=0
                    avg_l+=0
            area=trimmeddic[k][0][2]
            name=trimmeddic[k][0][0]
            bprices=avg_b/len(trimmeddic[k])
            lprices=avg_l/len(trimmeddic[k])
            conf=trimmeddic[k][0][1]
            lat=trimmeddic[k][0][5]
            longi=trimmeddic[k][0][6]
            listdate=trimmeddic[k][0][7]
            txn_types=trimmeddic[k][0][8]
            spamwriter.writerow([name,conf,area,bprices,lprices,lat,longi,listdate,txn_types,listing])


# with open('output_lat.csv', 'wb') as csvfile:
#     spamwriter = csv.writer(csvfile, delimiter=' ',
#                             quotechar=',', quoting=csv.QUOTE_MINIMAL)
    
#     for k in trimmeddic:
#         if len(trimmeddic[k])==1:
#             price=trimmeddic[k][0][2]
#             lat=trimmeddic[k][0][3]
#             longi=trimmeddic[k][0][4]
#             spamwriter.writerow([lat,longi,price])
#         # elif trimmeddic[k][0][0]!=trimmeddic[k][1][0]:
#         #     for t in range(0,len(trimmeddic[k])):
#         #         spamwriter.writerow(trimmeddic[k][t])
#         else:
#             avg=0.0
#             for t in range(0,len(trimmeddic[k])):
#                 #print (trimmeddic[k][t][2])
#                 try:
#                     avg+=float(trimmeddic[k][t][2])
                    
#                 except ValueError:
#                     print (trimmeddic[k][t][2])
#                     avg+=0
                
#             print("sum ghe")
#             print(avg)
#             #name=trimmeddic[k][0][0]
#             price=avg/len(trimmeddic[k])
#             #conf=trimmeddic[k][0][1]
#             lat=trimmeddic[k][0][3]
#             longi=trimmeddic[k][0][4]
#             #print(price)
#             spamwriter.writerow([lat,longi,price])
    
    
