#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[7]:


df=pd.read_csv("C:/Users/vknsr/OneDrive/Documents/ML/weight-height.csv")
df.pop("Weight")
df


# In[8]:


plt.hist(df.Height,bins=20,rwidth=0.5)
plt.show()


# In[9]:


plt.scatter(df.Height,df.Gender,color="r")


# In[10]:


df.isnull().sum()
df.describe()


# In[11]:


lower_limit=df.Height.mean()-df.Height.std()*2
lower_limit


# In[12]:


upper_limit=df.Height.mean()+df.Height.std()*2
upper_limit


# In[13]:


df1=df[(df.Height>=lower_limit) & (df.Height<=upper_limit)]


# In[14]:


df1


# In[15]:


df1.describe()


# In[16]:


plt.hist(df1.Height,bins=20,rwidth=0.9)
plt.show()


# Z-Score Method

# In[17]:


df['z_score']=(df.Height-df.Height.mean())/df.Height.std()


# In[18]:


df['z_score']


# In[19]:


df.head()


# In[20]:


df[(df.z_score<=2)&(df.z_score>=-2)]


# In[ ]:




