#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


df=pd.read_csv("C:/Users/vknsr/Downloads/Iris.csv")
df = df.set_index('Id')


# In[4]:


df.head()


# In[5]:


df.describe()


# In[7]:


x=df.iloc[:,1:2]
y=df.iloc[:,4]


# In[97]:


from sklearn.model_selection import train_test_split 


# In[98]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[99]:


from sklearn.preprocessing import StandardScaler    
st_x= StandardScaler()  
x_train= st_x.fit_transform(x_train)    
x_test= st_x.transform(x_test) 


# In[100]:


from sklearn.tree import DecisionTreeClassifier


# In[101]:


classifier= DecisionTreeClassifier(criterion='entropy', random_state=0)  


# In[102]:


classifier.fit(x_train,y_train)


# In[103]:


y_pred= classifier.predict(x_test) 


# In[104]:


y_pred


# In[105]:


from sklearn import tree


# In[107]:


print(tree.plot_tree(classifier,filled=True))


# In[108]:


from sklearn.metrics import accuracy_score


# In[109]:


accuracy_score(y_test,y_pred)


# In[ ]:




