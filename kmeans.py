import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
veriler=pd.read_csv('musteriler.csv')
#print(veriler)
X= veriler.iloc[:,3:].values
#Kmeans
from sklearn.cluster import KMeans
k_means=KMeans(n_clusters= 3,init='k-means++')
k_means.fit(X)
print(k_means.cluster_centers_)
sonuçlar=[]
for i in range(1,11):
    k_means=KMeans(n_clusters=i ,init='k-means++',random_state=123)
    k_means.fit(X)
    sonuçlar.append(k_means.inertia_)
   # print(sonuçlar)
plt.plot(range(1,11),sonuçlar)
plt.title('KMeans')
plt.show()
#hc
from sklearn.cluster import AgglomerativeClustering

ag=AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='ward')
y_tahmin=ag.fit_predict(X)
print(y_tahmin)
plt.scatter(X[y_tahmin==0,0],X[y_tahmin==0,1])
plt.scatter(X[y_tahmin==1,0],X[y_tahmin==1,1])
plt.scatter(X[y_tahmin==2,0],X[y_tahmin==2,1])
plt.title('HC')
plt.show()
import scipy.cluster.hierarchy as sch
dengrogram=sch.dendrogram(sch.linkage(X,method='ward'))
plt.show()