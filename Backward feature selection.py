#!/usr/bin/env python
# coding: utf-8

# In[81]:


import pandas as pd
import numpy as np


# In[82]:


df=pd.read_csv("C:/Users/vknsr/OneDrive/Documents/ML/Titanic.csv")
df.head()
df.columns


# In[83]:


df.head()
df['Age'].fillna(value=df['Age'].mean(),inplace=True)
df['Cabin'].fillna(value=df['Cabin'].mode()[0],inplace=True)
df['Embarked'].fillna(value=df['Embarked'].mode()[0],inplace=True)


# In[84]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df["Survived"]=le.fit_transform(df["Survived"])
df["Name"]=le.fit_transform(df["Name"])
df["Sex"]=le.fit_transform(df["Sex"])
df["Ticket"]=le.fit_transform(df["Ticket"])
df["Cabin"]=le.fit_transform(df["Cabin"])
df["Embarked"]=le.fit_transform(df["Embarked"])


# In[85]:


from sklearn.model_selection import train_test_split


# In[86]:


y = df['Survived']
x = df.drop('Survived',axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[87]:


from sklearn.ensemble import RandomForestClassifier 
rf=RandomForestClassifier()


# In[88]:


from mlxtend.feature_selection import SequentialFeatureSelector 


# In[95]:


sfs=SequentialFeatureSelector(rf,k_features=10,forward=False,verbose=2,cv=5)


# In[96]:


sfs.fit(x_train,y_train)


# In[97]:


pd.DataFrame.from_dict(sfs.get_metric_dict()).T


# In[100]:


sfs.k_score_


# In[ ]:




