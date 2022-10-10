#!/usr/bin/env python
# coding: utf-8

# In[126]:


import pandas as pd
import numpy as np


# In[127]:


df=pd.read_csv("C:/Users/vknsr/OneDrive/Documents/ML/Titanic.csv")
df.head()


# In[128]:


df.drop('Name',axis=1,inplace=True)
df.drop('Cabin',axis=1,inplace=True)
df.drop('Ticket',axis=1,inplace=True)
df.drop('Embarked',axis=1,inplace=True)


# In[129]:


df.head()


# In[130]:


df['Age'].fillna(df['Age'].mean())


# In[131]:


Sex=pd.get_dummies(df['Sex'])
df=pd.concat([df,Sex],axis=1)


# In[132]:


df.head()


# In[133]:


df.drop('Sex',axis=1,inplace=True)


# In[134]:


df.isnull().sum()


# In[135]:


age_mean=df['Age'].mean()


# In[136]:


df['Age'].fillna(value=age_mean,inplace=True)


# In[137]:


df.isnull().sum()


# In[138]:


from sklearn.model_selection import train_test_split


# In[139]:


x=df.drop('Survived',axis=1)


# In[140]:


y=df['Survived']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[141]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()


# In[142]:


knn.fit(x_train,y_train)


# In[143]:


y_pred=knn.predict(x_test)


# In[144]:


from sklearn.metrics import accuracy_score


# In[145]:


accuracy_score(y_test,y_pred)


# In[157]:


error_rate=[]
for i in range(1,30):
    knn1=KNeighborsClassifier(n_neighbors=i)
    knn1.fit(x_train,y_train)
    pred_i=knn1.predict(x_test)
    error_rate.append(np.mean(pred_i!=y_test))


# In[158]:


import matplotlib.pyplot as plt


# In[159]:


plt.figure(figsize=(7,5))
plt.plot(range(1,30),error_rate)


# In[162]:


error_rate


# In[194]:


knn2=KNeighborsClassifier(n_neighbors=20)
knn2.fit(x_train,y_train)


# In[195]:


y_pred2=knn2.predict(x_test)
accuracy_score(y_test,y_pred)


# In[196]:


plt.figure(figsize=(10,7))
plt.plot(range(1,30),error_rate)


# In[ ]:




