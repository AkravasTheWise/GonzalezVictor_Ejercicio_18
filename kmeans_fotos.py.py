#!/usr/bin/env python
# coding: utf-8

# In[66]:


import glob
import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster


# In[52]:


archivos=glob.glob('*.png')


# In[58]:


data=[]
for archivo in archivos:
    imagen=plt.imread(archivo)
    img=imagen.reshape((-1,3))
    data.append(img)

data=np.array(data)
data=data.reshape((87,-1))
print(np.shape(data))


# In[61]:


inertia=np.zeros((20,2))
for n_clusters in range(20):
    k_means = sklearn.cluster.KMeans(n_clusters=n_clusters+1)
    k_means.fit(data)
    cluster = k_means.predict(data)
    inertia[n_clusters]=[n_clusters+1,k_means.inertia_]
    print(100*(n_clusters+1)/20,'%',' ',n_clusters+1,' clusters procesados')
    
#    X_centered = X.copy()
#    for i in range(n_clusters):
#        ii = cluster==i
#        X_centered[ii,:] = np.int_(k_means.cluster_centers_[i])


# In[67]:


plt.plot(inertia[:,0],inertia[:,1])
plt.ylabel('Inercia')
plt.xlabel('Número de clusters')
plt.savefig('inercia.png')
plt.title('El número óptimo de clusters es K=4')


# In[139]:


n_clusters=5
plt.figure(figsize=(8,8))
j=0
k_means = sklearn.cluster.KMeans(n_clusters=n_clusters)
k_means.fit(data)
cluster = k_means.predict(data)
norma=k_means.transform(data)
for i in range(n_clusters):

    archiv=np.array(archivos)
    archiv=archiv[np.argsort(norma[:,i])[-5:]]
    
    
    for archivo in archiv:
        plt.subplot(4,5,j+1)
        img=plt.imread(archivo)
        plt.imshow(img)
        j+=1
plt.tight_layout()
plt.savefig('ejemplo_clases.png')

