import datetime
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans2, whiten
import sys
import csv
import numpy as np
from nltk.cluster import KMeansClusterer, GAAClusterer, euclidean_distance
import nltk.corpus
from nltk import decorators
import nltk.stem
import re

stemmer_func = nltk.stem.snowball.EnglishStemmer().stem
stopwords = set(nltk.corpus.stopwords.words('english'))


kmeans2_n_clusters=5
kmeans2_n_clusters_latlong=50

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
    return np.array([
        word in title_components and not word in stopwords
        for word in words], np.short)

if __name__ == '__main__':
 with open('Jagaha_Items.csv', 'rb') as f:
        reader = csv.reader(f)
        your_list = list(reader)

        str_no_cluster=10
        job_titles=[]
        print '-----------------------------Stage 1----------------------------------------------------------------'
        print ' '
        clust_list=[]
        c=1
        count=0
        ccount=0
        t=0 
        x="" 
        for i in your_list:
            if(i[4]!=""):
                if re.search('[0-9a-zA-Z0-9]',i[4]):
                    if "Lease" in i[20]:
                                x = str(i[16]) 
                                x= x[4:-10]
                                x=x.replace(',','')
                                clust_list.append([str(i[4]),str(i[3]),float(x),float(i[12]),float(i[13])])
                    elif 'Buy' in i[19]:
                        x =str(i[18])
                        x=x.replace(',','') 
                        clust_list.append([str(i[4]),str(i[3]),float(x[4:]),float(i[12]),float(i[13])])
                    job_titles.append(str(i[4]))

            else:
                job_titles.append("Empty")
    	        ccount=ccount+1
        words = get_words(job_titles)
        cluster = KMeansClusterer(str_no_cluster, euclidean_distance,avoid_empty_clusters=True)
        dic={}
        clust_list_2=[]
        clust_list_final=[]
        for temp in clust_list:
            for j in temp[0]:
                if "Towers" or "Tower" or "Apartments" or "Apartment"in j:
                    clust_list_final.append(j.replace("Towers"or "Tower" or "Apartments" or "Apartment",""))
                else:
                    clust_list_final.append(j)

        for title in clust_list_final:
            clust_list_2.append(vectorspaced(title))
        cluster.cluster(clust_list_2)

        classified_examples = [
                cluster.classify(vectorspaced(title)) for title in clust_list_final
            ]
        i=0
        while(i<(str_no_cluster)):
          dic.update({i:[]})
          i+=1


        for cluster_id, title in sorted(zip(classified_examples, clust_list)):
            dic[int(cluster_id)].append([title[0],title[1],title[2],title[3],title[4]])
            

        print ' -------------------------Stage 1 with empty clusters-------------------------------------'
        print '  '

        for a in dic:
            print"  dictionary "
            print dic[a]       

        for x in list(dic.keys()):
            if dic[x]==[]:
                del dic[x]
        
        cleaned_names_dic={}  
        keys=0
        new_key=0
        
        for key,value in dic.items():
            new_key=keys
            cleaned_names_dic[new_key]=dic[key]
            keys=keys+1
        
        
        print ' -------------------------Stage 1 FINAL clusters-------------------------------------'
        print '  '
        
        for a in cleaned_names_dic:
            print cleaned_names_dic[a]
            print ' '

        print"  ************************* bhk clustering *********************** "
        global_dic={}
        i=0
        while(i<(str_no_cluster*kmeans2_n_clusters)):
          global_dic.update({i:[]})
          i+=1

        count=0
        kiti=0
        for i in cleaned_names_dic:
            cluster_bhk_list=[]
            c=0                     #traversing 1 particular item in 1 list inside the dictionary
            for j in cleaned_names_dic[i]:
                bhk_str=j[1]
                if not re.search('[0-9]',bhk_str[:1]):
                    j[1]="1 BHK"
                bhk_str=j[1]    
                new_bhk=bhk_str[:1]
                a=float(new_bhk)
                cluster_bhk_list.append(a)
            # print 'because they say it is empty'
            # print ' '
            # print cluster_bhk_list    
            # print ' '
            cluster_bhk_numpy=np.array(cluster_bhk_list)
            #print 'before xy -----------------'+str(kiti)
            if len(cluster_bhk_list)>1:
                x,y = kmeans2(whiten(cluster_bhk_numpy), kmeans2_n_clusters, iter = 20) 
             #   print 'after xy ------------------------'+str(kiti)
             #   print y
              #  print y[0]
                kiti+=1
                u=0
                print("here")
                print (y)
                for u in y:
                    k=int(u)
            #        print 'k cartoy print' + str(k)
                    global_dic[k+count].append(cleaned_names_dic[i][c])
                    c+=1
            else:
                global_dic[0+count].append(cleaned_names_dic[i][c])
            count+=kmeans2_n_clusters        

        print"  --------------- bhk clusters------------------------"
       
            #initialie global dictionary    
        for x in list(global_dic.keys()):
            if global_dic[x]==[]:
                del global_dic[x]  
        
        new_global_dic={}
  

        for i in global_dic:
            print "  "
            print "  --------------- cluster "+str(i)+"------------------------"
            for k in global_dic[i]:
                print k[1]

        trimmeddic={}
        while(i<(len(global_dic))):
            trimmeddic.update({i:[]})
            i+=1

        print '     '    
        print 'trimed -----------'
        keys=0
        new_key=0
        for key,value in global_dic.items():
            new_key=keys
            trimmeddic[new_key]=global_dic[key]
            keys=keys+1

        print '     '    
        print "the STAGE 2 FINAL is herer########################################################"
        for k,v in trimmeddic.items():
            print k
            print v
            print " "



        
        print ' '
        print '++++++++++++++++++++++++++++++  Stage 3    +++++++++++++++++++++++'
        print ' '
        i=0
        while(i<(str_no_cluster*kmeans2_n_clusters_latlong*kmeans2_n_clusters)):
            new_global_dic.update({i:[]})
            i+=1

        count=0
        for i in trimmeddic:
            cluster_latlong_list=[]
            c=0
            #created temporary list within Each Cluster for Lat-long to be provided for Kmeans2

            for j in trimmeddic[i]:
                cluster_latlong_list.append([float(j[3]),float(j[4])])


            cluster_latlong_numpy=np.array(cluster_latlong_list)
            digits=cluster_latlong_numpy
            if len(cluster_latlong_list)>1:
                try:
                    x,y = kmeans2(whiten(digits), kmeans2_n_clusters_latlong, iter = 20)                 
                except:
                #    print 'in here '
                    print cluster_latlong_list
                    #cluster_latlong_list[0][0]=cluster_latlong_list[0][0]+0.01
                    #cluster_latlong_list[0][1]=cluster_latlong_list[0][1]+0.01
                    #digits=np.array(cluster_latlong_list)
                    #print np.all(np.linalg.eigvalsh(digits))
                    x,y = kmeans2(whiten(digits), kmeans2_n_clusters_latlong, iter = 20) 


                
                u=0
                for u in y:
                    k=int(u)
           #         print 'k cartoy print' + str(k)
                    new_global_dic[k+count].append(trimmeddic[i][c])
                    c+=1
            else:
                new_global_dic[0+count].append(trimmeddic[i][c])        
            count+=kmeans2_n_clusters_latlong

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

        print '  '
        print "--------------------STAGE 3 FINALlat-long clustering(*****************************************************************"    
        for i in final_trimmed_dic:
            print " count"
            print " "
            print final_trimmed_dic[i]
            
            
      