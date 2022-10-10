#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[19]:


data={'city':['chennai','bangalore','delhi','chennai','bangalore','delhi','chennai','bangalore','delhi','chennai','bangalore','delhi'],'Temp':[92,88,84,79,72,95,79,86,90,89,79,96],'event':['sunny','rainy','cloudy','rainy','rainy','rainy','sunny','cloudy','sunny','sunny','cloudy','sunny']}


# In[20]:


data1=pd.DataFrame(data)


# In[21]:


gb=data1.groupby('city')


# In[25]:


for city,city_df in gb:
    print(city)
    print(city_df)


# In[31]:


gb.mean()


# In[37]:


gb.max()


# In[33]:


gb.describe()


# In[41]:


gb.min()['Temp']


# In[39]:


conda install -c anaconda anaconda-navigator


# In[ ]:




