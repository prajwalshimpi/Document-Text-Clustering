import sys
import csv
import numpy
from nltk.cluster import KMeansClusterer, GAAClusterer, euclidean_distance
import nltk.corpus
from nltk import decorators
import nltk.stem
import re
# from pyspark import SparkContext,SparkConf
stemmer_func = nltk.stem.snowball.EnglishStemmer().stem
stopwords = set(nltk.corpus.stopwords.words('english'))

# conf = (SparkConf()
#          .setMaster("local[*]")
#          .setAppName("My app")
#          .set("spark.executor.memory", "1g"))
# sc = SparkContext(conf = conf)

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

if __name__ == '__main__':

    
 
 with open('makaan.csv', 'rb') as f:
        reader = csv.reader(f)
        your_list = list(reader)
        job_titles=[]
        clust_list=[]
        c=1
        count=0
        ccount=0
        t=0  
        for i in your_list:
        # if(c==1000):
        #     break
        # else:
            if(i[2]!=""):
                if re.search('[0-9a-zA-Z0-9]',i[2]):
                    #if "Towers" in i[2]:
                    clust_list.append(str(i[2]))
                    job_titles.append(str(i[2]))
                    c=c+1
                    # else:    
                    #     job_titles.append(str(i[2]))
                    #     c=c+1
            else:
                job_titles.append("Empty")
                ccount=ccount+1

        words = get_words(job_titles)
        print len(job_titles)
        print job_titles
        cluster = KMeansClusterer(5, euclidean_distance,avoid_empty_clusters=True)
        #cluster = GAAClusterer(5)
       
        clust_list_2=[]
        for temp in clust_list:
            if "Towers" or "Tower" or "Apartments" or "Apartment"in temp:
                temp=temp.replace("Towers","")
        for title in clust_list:
            if title:
                clust_list_2.append(vectorspaced(title))
        cluster.cluster(clust_list_2)

        # NOTE: This is inefficient, cluster.classify should really just be
        # called when you are classifying previously unseen examples!
        classified_examples = [
                cluster.classify(vectorspaced(title)) for title in clust_list
            ]

        for cluster_id, title in sorted(zip(classified_examples, your_list)):
            # if "Empty" in title:
            #     count=count+1
            print"   "
            print cluster_id,title[2],title[4]
        #     # t=t+1
            # print t
        # print(ccount)
        # print(count)