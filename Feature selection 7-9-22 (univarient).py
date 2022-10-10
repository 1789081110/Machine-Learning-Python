#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np


# In[11]:


df=pd.read_csv('C:/Users/vknsr/Downloads/mobiledata.csv')


# In[12]:


df.head()


# In[13]:


from sklearn.feature_selection import SelectKBest


# In[14]:


from sklearn.feature_selection import chi2


# In[15]:


df1=df.pop('price_range')


# In[16]:


best_feature=SelectKBest(score_func=chi2,k=10)


# In[17]:


fit=best_feature.fit(df,df1)


# In[20]:


fit.scores_ #chi2 value for each value


# In[21]:


df.columns


# In[22]:


dfscores=pd.DataFrame(fit.scores_)


# In[23]:


dfscores.head()


# In[24]:


dfspecification=pd.DataFrame(df.columns)


# In[27]:


featurescores=pd.concat([dfspecification,dfscores],axis=1)


# In[28]:


featurescores.columns=['specs','score']


# In[29]:


featurescores


# In[31]:


dfnew=featurescores.nlargest(10,'score')


# In[32]:


dfnew


# In[ ]:




