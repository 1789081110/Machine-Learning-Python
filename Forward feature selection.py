#!/usr/bin/env python
# coding: utf-8

# In[38]:


get_ipython().system('pip install mlxtend')


# In[39]:


import pandas as pd
import numpy as np


# In[53]:


df=pd.read_csv("C:/Users/vknsr/Downloads/wine.csv")
df.head()


# In[41]:


df.info()


# In[42]:


df1=df.pop('Wine')


# In[43]:


from sklearn.model_selection import train_test_split


# In[44]:


x_train,x_test,y_train,y_test=train_test_split(df,df1,test_size=0.2)


# In[45]:


from sklearn.ensemble import RandomForestClassifier 
rf=RandomForestClassifier()


# In[46]:


rf.fit(x_train,y_train)


# In[47]:


from mlxtend.feature_selection import SequentialFeatureSelector 


# In[48]:


sfs=SequentialFeatureSelector(rf,k_features=6,forward=True,verbose=2,cv=5)


# In[49]:


sfs.fit(x_train,y_train)


# In[50]:


pd.DataFrame.from_dict(sfs.get_metric_dict()).T


# In[52]:


sfs.k_score_


# In[ ]:




