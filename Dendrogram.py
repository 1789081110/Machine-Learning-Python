#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd


# In[4]:


import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


# In[5]:


df=make_blobs(n_samples=200,n_features=2,centers=4,cluster_std=1.6,random_state=50)


# In[7]:


df


# In[8]:


points=df[0]


# In[12]:


import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering


# In[13]:


dendrogram= sch.dendrogram(sch.linkage(points,method='ward'))


# In[ ]:




