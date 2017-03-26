#to cluster time
import datetime
from sklearn.cluster import KMeans
import numpy as np
import random
import csv
def calc_date(st):
    today=datetime.date.today()
    a=(st.split('T',2))[0].split('-',3)
    b=[]
    for i in a:
        b.append(int(i))
    calc=datetime.date(b[0],b[1],b[2])
    d=today-calc
    return d.days
def mean(lis):
    if len(lis)>0:
        a=sum(lis)/len(lis)
    else:
        a=0
    return a
tim=[]
nam=[]
day=[]
monthold=[]
weekold=[]
new=[]
dic={}

data={}
c=0
with open('Purpleyo_Items.csv','rb') as csvfile:
    reader=csv.reader(csvfile,)
    for row in reader:
        tim.append(row[7])
        nam.append(row)
    del tim[0],nam[0]
    for i in tim:
        day.append(calc_date(i))
    for i in day:
        c+=1
    days=np.array(day)
    print len(days)

    d=KMeans()
    d.fit(days.reshape(-1,1))
    a=int(d.n_clusters)
    i=0
    while i<a:
        dic.update({i:[]})
        i+=1
    print dic
    for i in day:
        b=d.predict(i)
        b=int(b)
        dic[b].append(i)
    while i>-1:
        print i,dic[i]
        i-=1







    for i in global_dic:
        job_titles=[]
        for j in global_dic[i]:
            if re.search('[0-9a-zA-Z0-9]',j[2]):
                job_titles.append(str(j[2]))    
        print "yaha"
        print job_titles
        words = get_words(job_titles)
        print "words yaha"
        print words
        cluster = KMeansClusterer(10, euclidean_distance)
            #cluster = GAAClusterer(5)
        cluster.cluster([vectorspaced(title) for title in job_titles if title])

            # NOTE: This is inefficient, cluster.classify should really just be
            # called when you are classifying previously unseen examples!
        classified_examples = [
            cluster.classify(vectorspaced(title)) for title in job_titles
            ]
        for cluster_id, title in sorted(zip(classified_examples, job_titles)):
            print cluster_id, title




        #export PATH=$PATH:/home/prajwal/Downloads/spark-1.6.1-bin-hadoop2.6/bin
# export PYTHONPATH=$SPARK_HOME/python/:$PYTHONPATH
# export PYTHONPATH=$SPARK_HOME/python/lib/py4j-0.8.2.1-src.zip:$PYTHONPATH
# export SPARK_HOME=/home/prajwal/Downloads/spark-1.6.1-bin-hadoop2.6
# export JAVA_HOME=JAVA_HOME=/usr/lib/jvm/java-1.8.0