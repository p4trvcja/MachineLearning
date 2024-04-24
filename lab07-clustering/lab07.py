#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import fetch_openml
import numpy as np
import pickle


# ## 2. Data preparation

# In[2]:


mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
mnist.target = mnist.target.astype(np.uint8)
X = mnist["data"]
y = mnist["target"]


# ## 3.1 Clustering

# In[3]:


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

silhouette_list = []

for k in range (8,13):
    kmeans = KMeans(n_clusters=k, n_init=10).fit(X)
    silhouette_list.append(silhouette_score(X, kmeans.labels_))

with open("kmeans_sil.pkl", 'wb') as f:
    pickle.dump(silhouette_list, f)


# In[4]:


from sklearn.metrics import confusion_matrix
km_10 = KMeans(n_clusters=10, n_init=10).fit(X)
y_pred_10 = km_10.predict(X)

conf_m = confusion_matrix(y, y_pred_10)


# ## 3.5

# In[5]:


indices_arr = np.argmax(conf_m, axis=1)
set_val = sorted(set(indices_arr))

with open("kmeans_argmax.pkl", 'wb') as f:
    pickle.dump(list(set_val), f)

print(set_val)


# ## 3.6

# In[6]:


distances = np.array([np.linalg.norm(X[i] - X[j]) for i in range(300) for j in range(len(X))])
distances = [i for i in distances if i != 0]

distances = np.sort(distances)[:10]

with open("dist.pkl", 'wb') as f:
    pickle.dump(list(distances), f)

print("distances = ", distances)


# ## 3.7

# In[7]:


s = np.mean(distances[:3])
eps_min = s
eps_max = s + 0.1 * s
eps_step = 0.04 * s

eps_values = np.arange(eps_min, eps_max, eps_step)
print(eps_values)


# ## 3.8

# In[ ]:


from sklearn.cluster import DBSCAN
labels = []

for eps in eps_values:
    dbscan = DBSCAN(eps=eps)
    dbscan.fit(X)
    n_labels = len(np.unique(dbscan.labels_))
    labels.append(n_labels)

with open("dbscan_len.pkl", "wb") as f:
    pickle.dump(labels, f)
print(labels)


# In[ ]:




