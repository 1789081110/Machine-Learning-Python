#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[9]:


df=pd.read_csv("C:/Users/vknsr/OneDrive/Documents/ML/Iris.csv")
df


# In[10]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[11]:


df["Species"]=le.fit_transform(df["Species"])
df["Species"]


# In[12]:


df1=df.pop("Species")
df1.head()


# In[13]:


from sklearn.model_selection import train_test_split


# In[14]:


x_train,x_test,y_train,y_test=train_test_split(df,df1,test_size=0.2)


# In[15]:


from sklearn.svm import SVC
svc=SVC()


# In[16]:


svc.fit(x_train,y_train)


# In[17]:


y_pred=svc.predict(x_test)


# In[18]:


from sklearn.metrics import accuracy_score


# In[23]:


accuracy_score(y_pred,y_test)


# In[ ]:




